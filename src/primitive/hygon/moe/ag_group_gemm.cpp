#include "moe_core.h"
#include "moe_assets.h"
#include "hip/hip_runtime.h"
#include "hip_kernels/ag_group_gemm_kernels.h"
#include "hip_kernels/helper_kernels.h"
#include "moe_core/ag_group_gemm.h"
#include "moe_core/runtime_utils.h"
#include "ag_default_environment.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <dlfcn.h>
#include <filesystem>
#include <exception>
#include <functional>
#include <thread>
#include <utility>
#include <unordered_set>
#include <vector>

namespace ftrain
{
namespace moe
{
namespace core
{

namespace
{

void check_supported_dtype(ScalarType dtype)
{
    FTRAIN_CHECK(dtype == ScalarType::Half ||
               dtype == ScalarType::BFloat16)
        << "AGGroupGemm supports FP16/BF16 input and output";
}

void all_ranks_sync(hipStream_t stream,
                    std::optional<std::vector<Tensor>> sync_buffers,
                    std::optional<int> rank)
{
    if (get_bool_from_env("FTRAIN_BARRIER_ALL_ON_STREAM", true))
    {
#ifdef FTRAIN_SHM_USE_NVSHMEM
        barrier_all_on_stream(stream);
#else
        barrier_all_on_stream(stream, sync_buffers, rank);
#endif
    }
}

int get_int_from_env_local(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
    {
        return default_value;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value)
    {
        return default_value;
    }
    return static_cast<int>(parsed);
}

int get_group_n_chunk_size_local()
{
    int value = get_int_from_env_local(
        "FTRAIN_AG_GROUP_GEMM_ASM_V36_GROUP_N_CHUNK", 256);
    if (value < 256 || value % 256 != 0)
    {
        value = 256;
    }
    return value;
}

const char *get_env_with_legacy_local(const char *name,
                                      const char *legacy_name)
{
    const char *value = std::getenv(name);
    return value != nullptr ? value : std::getenv(legacy_name);
}

bool get_bool_from_env_with_legacy_local(const char *name,
                                         const char *legacy_name,
                                         bool default_value)
{
    const char *value = get_env_with_legacy_local(name, legacy_name);
    if (value == nullptr)
    {
        return default_value;
    }
    return std::string(value) == "1" || std::string(value) == "ON";
}

int get_int_from_env_with_legacy_local(const char *name,
                                       const char *legacy_name,
                                       int default_value)
{
    const char *value = get_env_with_legacy_local(name, legacy_name);
    if (value == nullptr || value[0] == '\0')
    {
        return default_value;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end == value ? default_value : static_cast<int>(parsed);
}

class AGWeightPackStream
{
public:
    AGWeightPackStream() = default;
    AGWeightPackStream(const AGWeightPackStream &) = delete;
    AGWeightPackStream &operator=(const AGWeightPackStream &) = delete;

    ~AGWeightPackStream()
    {
        if (stream_ != nullptr)
        {
            int previous_device = device_;
            hipGetDevice(&previous_device);
            hipSetDevice(device_);
            hipStreamDestroy(stream_);
            hipSetDevice(previous_device);
        }
    }

    void initialize()
    {
        if (stream_ != nullptr)
            return;
        int least_priority = 0;
        int greatest_priority = 0;
        CUDA_CHECK(hipGetDevice(&device_));
        CUDA_CHECK(hipDeviceGetStreamPriorityRange(&least_priority,
                                                   &greatest_priority));
        CUDA_CHECK(hipStreamCreateWithPriority(&stream_, hipStreamNonBlocking,
                                               least_priority));
    }

    hipStream_t get() const { return stream_; }

private:
    hipStream_t stream_ = nullptr;
    int device_ = 0;
};

class ScopedRaggedRouteEnvironment
{
public:
    explicit ScopedRaggedRouteEnvironment(bool chunk_ready = false)
    {
        set("FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_ROUTED_INPUT", "1");
        set("FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE", "1");
        set("FTRAIN_AG_GROUP_GEMM_STRICT_RANK_TILE", "1");
        set("FTRAIN_AG_GROUP_GEMM_STABLE_EXPERT_RANK_SORT", "1");
        set("FTRAIN_AG_GROUP_GEMM_V40_TRUE_RING", "1");
        set("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY", "1");
        set("FTRAIN_AG_GROUP_GEMM_V40_BOOTSTRAP_STAGES", "0");
        set("FTRAIN_AG_GROUP_GEMM_PUSH_EARLY_REMOTE", "1");
        set("FTRAIN_AG_GROUP_GEMM_PUSH_PEER_API", "1");
        set("FTRAIN_AG_GROUP_GEMM_V11_PUSH_STREAM_PRIORITY", "-1");
        set("FTRAIN_AG_GROUP_GEMM_FORCE_RAGGED_M", "1");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_RING_CHUNK", chunk_ready ? "1" : "0");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_RING_OWNER_SIGNALS", "1");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_RING_WAIT_REMOTE_READY", "0");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_MODE", "3");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_GROUP_SIZE", "1");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_STRIDE_WORDS", "1024");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_GRID", "16");
        set("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", chunk_ready ? "1" : "0");
        set("FTRAIN_AG_GROUP_GEMM_NATIVE_KN_STAGE_SIGNAL_MAP", "1");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_BODY_SLOT_BASE", "24");
        set("FTRAIN_AG_GROUP_GEMM_FORCE_PHYSICAL_SIGNAL_LAYOUT", "1");
        set("FTRAIN_AG_GROUP_GEMM_PUSH_RING_GATE", "0");
        set("FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_CHUNK_ACQUIRE", "0");
    }

    ~ScopedRaggedRouteEnvironment()
    {
        for (auto it = old_.rbegin(); it != old_.rend(); ++it)
        {
            if (it->second.empty())
                unsetenv(it->first.c_str());
            else
                setenv(it->first.c_str(), it->second.c_str(), 1);
        }
    }

private:
    void set(const char *name, const char *value)
    {
        const char *old = std::getenv(name);
        old_.emplace_back(name, old == nullptr ? "" : old);
        setenv(name, value, 1);
    }

    std::vector<std::pair<std::string, std::string>> old_;
};

bool selected_mt256_co_contains_local(const char *token)
{
    const char *co = std::getenv("FTRAIN_AG_GROUP_GEMM_ASM_MT256_CO");
    return co != nullptr && std::strstr(co, token) != nullptr;
}

enum class AGRouteKind
{
    Auto = 0,
    NativeKn,
    AgNnA2a,
    NativeEnkPacket4Native,
};

struct AGRouteSpec
{
    AGRouteKind kind = AGRouteKind::Auto;
    bool native_kn = false;
    bool enk = false;
    bool nk_new_a2a = false;
    bool nk_new_a2a_chunk = false;
    bool nk_new_a2a_ring = false;
    bool nk_32k = false;
    bool enk_ring_chunk_ready = false;
    bool kn_ring_chunk_ready = false;
    std::string code_object;
    std::string code_object_func;
};

AGRouteSpec resolve_ag_route(const std::string &route)
{
    AGRouteSpec spec;
    std::string value = route;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (value == "auto" || value.empty())
    {
        return spec;
    }
    if (value == "ag_nn_a2a")
    {
        spec.kind = AGRouteKind::AgNnA2a;
    }
    else if (value == "ag_nt_a2a")
    {
        spec.kind = AGRouteKind::NativeEnkPacket4Native;
        spec.enk = true;
        spec.nk_new_a2a = true;
    }
    else if (value == "ag_nt_ring_chunk" ||
             value == "ag_nt_ring_chunk_32k")
    {
        spec.kind = AGRouteKind::NativeEnkPacket4Native;
        spec.enk = true;
        spec.nk_new_a2a_ring = true;
        spec.nk_32k = value == "ag_nt_ring_chunk_32k";
    }
    else if (value == "ag_nt_a2a_chunk" ||
             value == "ag_nt_a2a_chunk_32k")
    {
        spec.kind = AGRouteKind::NativeEnkPacket4Native;
        spec.enk = true;
        spec.nk_new_a2a_chunk = true;
        spec.nk_32k = value == "ag_nt_a2a_chunk_32k";
    }
    else if (value == "native_kn_megatron_ragged" ||
             value == "ag_nn_ring" ||
             value == "expanded_ring_ragged" ||
             value == "native_kn_megatron_ragged_ring")
    {
        spec.kind = AGRouteKind::NativeKn;
        spec.native_kn = true;
    }
    else if (value == "ag_nn_ring_chunk")
    {
        spec.kind = AGRouteKind::NativeKn;
        spec.native_kn = true;
        spec.kn_ring_chunk_ready = true;
    }
    else
    {
        FTRAIN_CHECK(false)
            << "Unsupported AGGroupGemm route: " << route
            << " (canonical: ag_nn_a2a, "
               "ag_nt_a2a, "
               "ag_nt_a2a_chunk, "
               "ag_nt_ring_chunk, "
               "ag_nt_a2a_chunk_32k, ag_nt_ring_chunk_32k, "
               "ag_nn_ring, ag_nn_ring_chunk)";
    }
    return spec;
}

void find_route_code_object(AGRouteSpec &spec)
{
    const std::string prefix =
        "ftrain_ag_grouped_gemm_contract_rank_tile_x64_";
    std::string preferred;
    std::string legacy;
    if (spec.kind == AGRouteKind::AgNnA2a)
    {
        preferred = prefix +
            "ag_nn_a2a_direct_output_nchunk_signal.co";
    }
    else if (spec.kind == AGRouteKind::NativeEnkPacket4Native)
    {
        if (spec.nk_32k)
        {
            preferred = prefix +
                (spec.nk_new_a2a_chunk
                    ? "ag_nt_a2a_chunk_32k_direct_output_nchunk_signal.co"
                    : "ag_nt_ring_chunk_32k_direct_output_nchunk_signal.co");
        }
        else
        {
            preferred = prefix +
                (spec.nk_new_a2a_chunk
                     ? "ag_nt_a2a_chunk_direct_output_nchunk_signal.co"
                     : spec.nk_new_a2a_ring
                     ? (spec.enk_ring_chunk_ready
                            ? "ag_nt_ring_chunk_direct_output_nchunk_signal.co"
                            : "ag_nt_ring_direct_output_nchunk_signal.co")
                     : "ag_nt_a2a_direct_output_nchunk_signal.co");
        }
    }
    else if (spec.kn_ring_chunk_ready)
    {
        preferred = prefix +
            "ag_nn_ring_chunk_nn_ailk_routed_direct_output_nchunk_signal.co";
    }
    else
    {
        preferred = prefix +
            "ag_nn_ring_routed_direct_output_nchunk_signal.co";
        legacy = prefix +
            "v36_dynamic_nn_ailk_routed_direct_output_nchunk_signal.co";
    }

    const auto preferred_path = ::ftrain::moe::assets::resolveCodeObject(preferred);
    std::string best;
    if (::ftrain::moe::assets::isCodeObjectAvailable(preferred_path))
    {
        best = preferred_path;
    }
    else if (!legacy.empty())
    {
        const auto legacy_path = ::ftrain::moe::assets::resolveCodeObject(legacy);
        if (::ftrain::moe::assets::isCodeObjectAvailable(legacy_path))
        {
            best = legacy_path;
        }
    }
    FTRAIN_CHECK(!best.empty())
        << "No default code object found for AGGroupGemm route";
    spec.code_object = best;
    spec.code_object_func = std::filesystem::path(best).stem().string();
}

class ScopedRouteEnvironment
{
public:
    explicit ScopedRouteEnvironment(const AGRouteSpec &spec)
        : spec_(spec)
    {
        set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", "1");
        set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", "1");
        set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_WGM8", "1");
        set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", "0");
        set("FTRAIN_AG_GROUP_GEMM_ASM_MT256", "1");
        set("FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE", "1");
        set("FTRAIN_AG_GROUP_GEMM_PUSH_EARLY_REMOTE", "1");

        if (spec_.kind == AGRouteKind::NativeEnkPacket4Native)
        {
            // ENK owns an independent direct consumer.  It uses the native-NK
            // launcher contract only for its [E,N,K] ABI; the code object and
            // symbol are route-specific and are never aliased to native_nk.
            set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT",
                "enk_ailk");
            set("FTRAIN_AG_GROUP_GEMM_INPUT_WEIGHT_NK", "1");
            set("FTRAIN_AG_GROUP_GEMM_FORCE_PHYSICAL_SIGNAL_LAYOUT", "1");
            set("FTRAIN_AG_GROUP_GEMM_PUSH_ZIGZAG_ORDER", "0");
            set("FTRAIN_AG_GROUP_GEMM_PUSH_PEER_API", "1");
            set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_CO",
                spec_.code_object.c_str());
            set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_FUNC",
                spec_.code_object_func.c_str());
        }
        else if (spec_.kind == AGRouteKind::AgNnA2a)
        {
            set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT", "kn_direct");
            set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_CO",
                spec_.code_object.c_str());
            set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_FUNC",
                spec_.code_object_func.c_str());
            // Variant 17 polls physical owner-rank barrier slots.  Stage-
            // signal layout was inherited from the v36 dynamic scheduler and
            // made a tile wait on an arrival-stage slot; when that slot was
            // published earlier than the tile's actual owner, the body could
            // read stale peer payload on gfx936.
            set("FTRAIN_AG_GROUP_GEMM_FORCE_PHYSICAL_SIGNAL_LAYOUT", "1");
            // Physical owner metadata is consumed in rank order; use the
            // matching push order instead of the stage/zigzag queue.
            set("FTRAIN_AG_GROUP_GEMM_PUSH_ZIGZAG_ORDER", "0");
            set("FTRAIN_AG_GROUP_GEMM_PUSH_PEER_API", "1");
            set("FTRAIN_AG_GROUP_GEMM_INPUT_WEIGHT_NK", "0");
            set("FTRAIN_AG_GROUP_GEMM_NATIVE_KN_INPUT_WEIGHT_NK", "0");
            set_if_absent("FTRAIN_AG_GROUP_GEMM_PUSH_MAX_INFLIGHT", "2");
        }
        else
        {
            set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT", "kn_direct");
            set("FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_ROUTED_INPUT", "1");
            set("FTRAIN_AG_GROUP_GEMM_STRICT_RANK_TILE", "1");
            set("FTRAIN_AG_GROUP_GEMM_STABLE_EXPERT_RANK_SORT", "1");
            set("FTRAIN_AG_GROUP_GEMM_V40_TRUE_RING", "1");
            set("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY", "1");
            set("FTRAIN_AG_GROUP_GEMM_V40_BOOTSTRAP_STAGES", "0");
            set("FTRAIN_AG_GROUP_GEMM_PUSH_PEER_API", "1");
            set("FTRAIN_AG_GROUP_GEMM_V11_PUSH_STREAM_PRIORITY", "-1");
            if (spec_.kn_ring_chunk_ready)
            {
                set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_CO", spec_.code_object.c_str());
                set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_FUNC", spec_.code_object_func.c_str());
            }
            if (!spec_.code_object.empty())
                set_if_absent("FTRAIN_AG_GROUP_GEMM_ASM_MT256_CO",
                              spec_.code_object);
            if (!spec_.code_object_func.empty())
                set_if_absent("FTRAIN_AG_GROUP_GEMM_ASM_MT256_FUNC",
                              spec_.code_object_func);
            ragged_.reset(new ScopedRaggedRouteEnvironment(
                spec_.kn_ring_chunk_ready));
        }

        if (spec_.kind == AGRouteKind::AgNnA2a ||
            spec_.kind == AGRouteKind::NativeEnkPacket4Native)
        {
            // These are direct/V36 contracts.  Isolate them from leaked
            // native-KN ring or routed-input state so route selection is
            // self-contained in production.
            unset("FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_ROUTED_INPUT");
            unset("FTRAIN_AG_GROUP_GEMM_STRICT_RANK_TILE");
            unset("FTRAIN_AG_GROUP_GEMM_STABLE_EXPERT_RANK_SORT");
            unset("FTRAIN_AG_GROUP_GEMM_V40_TRUE_RING");
            unset("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY");
            unset("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_PREFETCH_STAGES");
            unset("FTRAIN_AG_GROUP_GEMM_V40_BOOTSTRAP_STAGES");
            unset("FTRAIN_AG_GROUP_GEMM_FORCE_RAGGED_M");
            if (spec_.nk_new_a2a_ring)
            {
                set("FTRAIN_AG_GROUP_GEMM_PUSH_COPY", "1");
                set("FTRAIN_AG_GROUP_GEMM_PUSH_RING_GATE", "0");
                set_if_absent("FTRAIN_AG_GROUP_GEMM_V11_PUSH_STREAM_PRIORITY", "-1");
                set("FTRAIN_AG_GROUP_GEMM_RANK_TILE_DYNAMIC", "1");
                set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_V11_CHUNK_SIGNAL", "0");
                set("FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", "0");
                set("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", "0");
                const int owner_tile_mode = get_int_from_env_local(
                    "FTRAIN_AG_GROUP_GEMM_NK_NEW_RING_OWNER_TILES", 0);
                FTRAIN_CHECK(owner_tile_mode >= 0 && owner_tile_mode <= 3);
                if (owner_tile_mode != 0)
                {
                    set("FTRAIN_AG_GROUP_GEMM_STRICT_RANK_TILE",
                        owner_tile_mode == 1 ? "1" : "0");
                    set("FTRAIN_AG_GROUP_GEMM_SPLIT_LOCAL_RANK_TILE", "0");
                    set("FTRAIN_AG_GROUP_GEMM_STABLE_EXPERT_RANK_SORT", "1");
                }
                if (owner_tile_mode == 3)
                {
                    set("FTRAIN_AG_GROUP_GEMM_FORCE_PHYSICAL_SIGNAL_LAYOUT", "0");
                    set("FTRAIN_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT", "1");
                    set("FTRAIN_AG_GROUP_GEMM_PUSH_ZIGZAG_ORDER", "0");
                }
            }
            if (spec_.enk_ring_chunk_ready)
            {
                const bool history_schedule = get_bool_from_env("FTRAIN_ENK_RING_HISTORY_SCHEDULE", false);
                set("FTRAIN_AG_GROUP_GEMM_FORCE_PHYSICAL_SIGNAL_LAYOUT", history_schedule ? "0" : "1");
                set("FTRAIN_AG_GROUP_GEMM_NK_NEW_RING_OWNER_TILES", history_schedule ? "3" : "0");
                set("FTRAIN_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT", history_schedule ? "1" : "0");
                if (history_schedule)
                {
                    set("FTRAIN_AG_GROUP_GEMM_STRICT_RANK_TILE", "0");
                    set("FTRAIN_AG_GROUP_GEMM_SPLIT_LOCAL_RANK_TILE", "0");
                    set("FTRAIN_AG_GROUP_GEMM_STABLE_EXPERT_RANK_SORT", "1");
                }
                set("FTRAIN_AG_GROUP_GEMM_PUSH_ZIGZAG_ORDER", "0");
                set("FTRAIN_AG_GROUP_GEMM_PUSH_EARLY_REMOTE",
                    get_bool_from_env("FTRAIN_ENK_RING_REMOTE_BEFORE_LOCAL", false) ? "1" : "0");
                set("FTRAIN_AG_GROUP_GEMM_NK_NEW_RING_PREFETCH_STAGES", "0");
            }
            if (spec_.nk_new_a2a_chunk)
            {
                set("FTRAIN_AG_GROUP_GEMM_PUSH_COPY", "1");
                set("FTRAIN_AG_GROUP_GEMM_PUSH_RING_GATE", "0");
                set("FTRAIN_AG_GROUP_GEMM_RANK_TILE_DYNAMIC", "1");
                set("FTRAIN_AG_GROUP_GEMM_ASM_MT256_V11_CHUNK_SIGNAL", "0");
                set("FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", "0");
                set("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", "0");
                set("FTRAIN_AG_GROUP_GEMM_CHUNK_SIGNAL", "0");
                set("FTRAIN_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT", "0");
                set("FTRAIN_AG_GROUP_GEMM_BENCHMARK_PRELOADED_INPUT", "0");
            }
        }
    }

    ~ScopedRouteEnvironment()
    {
        ragged_.reset();
        for (auto it = old_.rbegin(); it != old_.rend(); ++it)
        {
            if (it->second.empty())
                unsetenv(it->first.c_str());
            else
                setenv(it->first.c_str(), it->second.c_str(), 1);
        }
    }

private:
    void set(const char *name, const char *value)
    {
        const char *old = std::getenv(name);
        old_.emplace_back(name, old == nullptr ? "" : old);
        setenv(name, value, 1);
    }

    void set_if_absent(const char *name, const std::string &value)
    {
        if (std::getenv(name) == nullptr)
            set(name, value.c_str());
    }

    void unset(const char *name)
    {
        const char *old = std::getenv(name);
        if (old == nullptr)
            return;
        old_.emplace_back(name, old);
        unsetenv(name);
    }

    AGRouteSpec spec_;
    std::unique_ptr<ScopedRaggedRouteEnvironment> ragged_;
    std::vector<std::pair<std::string, std::string>> old_;
};

} // namespace

__global__ void wait_signal_invalidate_kernel(const volatile int *signal,
                                               uint32_t signal_value)
{
    if (threadIdx.x == 0)
    {
        while (*signal < static_cast<int>(signal_value))
        {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    __builtin_amdgcn_buffer_wbinvl1_vol();
}

class AGGroupGemm::AGGroupGemmImpl
{
public:
    AGGroupGemmImpl(std::shared_ptr<Context> context,
                    int32_t nnodes,
                    int32_t full_m,
                    int32_t output_dim,
                    int32_t reduction_dim,
                    int32_t num_experts,
                    int32_t topk,
                    ScalarType input_dtype,
                    ScalarType output_dtype,
                    bool allocate_output_on_init,
                    std::vector<int64_t> rank_m_sizes,
                    int32_t capacity_full_m,
                    std::string route)
        : context_(std::move(context)), nnodes_(nnodes), full_m_(full_m),
          capacity_full_m_(std::max(
              full_m,
              capacity_full_m > 0
                  ? capacity_full_m
                  : get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_CAPACITY_FULL_M", full_m))),
          output_dim_(output_dim), reduction_dim_(reduction_dim),
          num_experts_(num_experts),
          topk_(topk), input_dtype_(input_dtype),
          output_dtype_(output_dtype),
          allocate_output_on_init_(allocate_output_on_init),
          world_size_(context_->world_size()), rank_(context_->rank()),
          local_world_size_(world_size_ / nnodes_),
          local_rank_(rank_ % local_world_size_), route_(route)
    {
        route_spec_ = resolve_ag_route(route_);
        ScopedAGDefaultEnvironment constructor_default_environment(
            route_spec_.kind == AGRouteKind::Auto);
        route_spec_.enk_ring_chunk_ready = route_spec_.nk_new_a2a_ring &&
            world_size_ == 8 && (route_spec_.nk_32k ||
                get_bool_from_env("FTRAIN_ENK_RING_CHUNK_READY", true));
        enk_ring_chunk_ready_ = route_spec_.enk_ring_chunk_ready;
        if (route_spec_.nk_32k)
        {
            FTRAIN_REQUIRE(get_device_name() == "gfx936" && world_size_ == 8 &&
                            topk_ == 1 && reduction_dim_ == 32768 &&
                            output_dim_ % 256 == 0 &&
                            input_dtype_ == ScalarType::BFloat16 &&
                            output_dtype_ == ScalarType::BFloat16,
                        route_, " requires gfx936, ETP8, BF16, K=32768, "
                        "N_local%256=0 and expanded input with operator topk=1");
        }
        std::unique_ptr<ScopedRouteEnvironment> constructor_route_guard;
        if (route_spec_.kind != AGRouteKind::Auto)
        {
            if ((route_spec_.native_kn || route_spec_.enk ||
                 route_spec_.kind == AGRouteKind::AgNnA2a) &&
                route_spec_.code_object.empty())
            {
                const bool packet4_large_address =
                    route_spec_.nk_new_a2a_chunk || route_spec_.nk_new_a2a_ring;
                const bool packet4_native_shape =
                    route_spec_.kind == AGRouteKind::NativeEnkPacket4Native &&
                    output_dim_ % 256 == 0 && reduction_dim_ % 64 == 0 &&
                    input_dtype_ == ScalarType::BFloat16 &&
                    (packet4_large_address ||
                     static_cast<uint64_t>(num_experts_) *
                            static_cast<uint64_t>(output_dim_) *
                            static_cast<uint64_t>(reduction_dim_) <=
                        (uint64_t{1} << 31));
                FTRAIN_REQUIRE(route_spec_.kind != AGRouteKind::NativeEnkPacket4Native ||
                                packet4_native_shape,
                            "ENK packet4 native LDS requires BF16, N%256=0, K%64=0",
                            packet4_large_address ? "" : " and weights <=4 GiB");
                if (packet4_large_address)
                {
                    FTRAIN_REQUIRE(
                        static_cast<uint64_t>(num_experts_) * output_dim_ <=
                            std::numeric_limits<uint32_t>::max() &&
                        static_cast<uint64_t>(reduction_dim_) <=
                            std::numeric_limits<uint32_t>::max() / (256 * sizeof(uint16_t)) &&
                        static_cast<uint64_t>(capacity_full_m_) <=
                            std::numeric_limits<uint32_t>::max() / sizeof(int32_t),
                        "ENK chunk/ring dimensions exceed per-tile or row-map address limits");
                }
                find_route_code_object(route_spec_);
            }
            constructor_route_guard.reset(
                new ScopedRouteEnvironment(route_spec_));
        }

        FTRAIN_CHECK(arch_check()) << "Unsupport arch:" << get_device_name();
        FTRAIN_CHECK(nnodes_ == 1)
            << "AGGroupGemm currently supports single-node ETP only";
        FTRAIN_CHECK(world_size_ % nnodes_ == 0)
            << "invalid nnodes/world_size";
        FTRAIN_CHECK(world_size_ <= MAX_LOCAL_WORLD_SIZE)
            << "AGGroupGemm IPC route supports at most MAX_LOCAL_WORLD_SIZE ranks";
        FTRAIN_CHECK(full_m_ > 0) << "full_m must be positive";
        FTRAIN_CHECK(capacity_full_m_ >= full_m_)
            << "capacity_full_m must be >= full_m";
        FTRAIN_CHECK(num_experts_ > 0) << "num_experts must be positive";
        FTRAIN_CHECK(topk_ > 0) << "topk must be positive";
        check_supported_dtype(input_dtype_);
        FTRAIN_CHECK(input_dtype_ == output_dtype_)
            << "AGGroupGemm expects output dtype == input dtype";

        if (route_spec_.kind == AGRouteKind::AgNnA2a)
        {
            FTRAIN_REQUIRE(get_device_name() == "gfx936" &&
                            input_dtype_ == ScalarType::BFloat16 &&
                            output_dim_ > 0 && output_dim_ % 256 == 0 &&
                            reduction_dim_ > 0 && reduction_dim_ % 64 == 0 &&
                            static_cast<uint64_t>(num_experts_) *
                                    static_cast<uint64_t>(output_dim_) <=
                                (uint64_t{1} << 31) /
                                    static_cast<uint64_t>(reduction_dim_) &&
                            topk_ == 1,
                        route_, " requires gfx936, BF16, N%256=0, "
                        "K%64=0, weights <=4 GiB and expanded expert-major "
                        "input with operator topk=1");
        }

        if (route_spec_.nk_new_a2a_ring)
        {
            FTRAIN_REQUIRE(get_device_name() == "gfx936" && topk_ == 1 &&
                            (world_size_ == 2 || world_size_ == 4 ||
                             world_size_ == 8),
                        route_, " requires gfx936, ETP2/4/8 and expanded "
                        "expert-major input with operator topk=1");
        }

        if (route_spec_.kn_ring_chunk_ready)
        {
            FTRAIN_REQUIRE(get_device_name() == "gfx936" &&
                            (world_size_ == 2 || world_size_ == 4 || world_size_ == 8) &&
                            topk_ == 1 && input_dtype_ == ScalarType::BFloat16 &&
                            output_dim_ % 256 == 0 && reduction_dim_ % 64 == 0,
                        route_, " requires gfx936, ETP2/4/8, BF16, N%256=0, "
                        "K%64=0 and expanded input with operator topk=1");
            signal_capacity_ = std::max(signal_capacity_, 256);
        }
        if (route_spec_.nk_new_a2a_chunk || enk_ring_chunk_ready_)
        {
            FTRAIN_REQUIRE(get_device_name() == "gfx936" && world_size_ == 8 &&
                            topk_ == 1 && (enk_ring_chunk_ready_ ||
                            supports_ragged_chunk_a2a() ||
                            (full_m_ % world_size_ == 0 && full_m_ / world_size_ >= 8)),
                        route_, " requires gfx936, ETP8 and expanded input with operator topk=1; "
                        "the legacy A2A chunk route additionally requires equal rank inputs of at least 8 rows");
            signal_capacity_ = std::max(signal_capacity_, 256);
        }
        if (route_spec_.kind == AGRouteKind::NativeKn)
        {
            FTRAIN_CHECK(!rank_m_sizes.empty())
                << "Native-KN ring routes require rank_m_sizes; "
                << "pass equal per-rank sizes for uniform ETP partitions";
            native_kn_ragged_ring_ = !rank_m_sizes.empty();
        }
        else
        {
            const bool routed_ragged_co_selected =
                selected_mt256_co_contains_local(
                    "nn_ailk_routed_direct_output_nchunk_signal");
            const bool chunk_alternative_requested =
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_CHUNK_ACQUIRE", false) ||
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", false) ||
                get_bool_from_env("FTRAIN_AG_GROUP_GEMM_RAGGED_RING_CHUNK", false);
            native_kn_ragged_ring_ = routed_ragged_co_selected &&
                !chunk_alternative_requested && !rank_m_sizes.empty();
        }

        if (world_size_ == 8 && get_device_name() == "gfx936" &&
            (route_spec_.nk_new_a2a ||
             (route_spec_.nk_new_a2a_chunk && !route_spec_.nk_32k)))
        {
            ag_group_gemm_system_fence_impl(context_->stream());
            CUDA_CHECK(hipStreamSynchronize(context_->stream()));
            weight_pack_stream_.initialize();
            ag_group_gemm_system_fence_impl(weight_pack_stream_.get());
            CUDA_CHECK(hipStreamSynchronize(weight_pack_stream_.get()));
        }

        CUDA_CHECK(hipMalloc(
            &d_rank_m_offsets_, (world_size_ + 1) * sizeof(int32_t)));
        set_rank_m_sizes(full_m_, std::move(rank_m_sizes));
        if (native_kn_ragged_ring_)
            ragged_m_ = true;

        if ((get_bool_from_env(
                 "FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", false) ||
             get_bool_from_env(
                 "FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", false)) &&
            std::getenv("FTRAIN_AG_GROUP_GEMM_SIGNAL_CAPACITY") == nullptr)
        {
            int signal_chunk_rows = get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_SIGNAL_CHUNK_ROWS", 128);
            signal_chunk_rows = std::max(signal_chunk_rows, 1);
            const int max_rank_rows =
                *std::max_element(rank_m_sizes_.begin(), rank_m_sizes_.end());
            const int chunks_per_rank =
                (max_rank_rows + signal_chunk_rows - 1) / signal_chunk_rows;
            const int required_signal_capacity =
                world_size_ * chunks_per_rank * 2;
            signal_capacity_ = std::max(
                signal_capacity_, std::min(required_signal_capacity, 4096));
        }

        input_buffers_ =
            create_tensor_list(
                {capacity_full_m_, reduction_dim_}, input_dtype_, context_);
        input_buffer_ = input_buffers_[local_rank_];
        void *input_ptrs[MAX_LOCAL_WORLD_SIZE] = {nullptr};
        for (int i = 0; i < world_size_; ++i)
        {
            input_ptrs[i] = input_buffers_[i].data_ptr();
        }
        CUDA_CHECK(
            hipMalloc(&d_input_ptrs_, MAX_LOCAL_WORLD_SIZE * sizeof(void *)));
        CUDA_CHECK(hipMemcpy(d_input_ptrs_,
                             input_ptrs,
                             MAX_LOCAL_WORLD_SIZE * sizeof(void *),
                             hipMemcpyHostToDevice));

        if (allocate_output_on_init_)
        {
            output_buffer_ = empty({capacity_full_m_ * topk_, output_dim_},
                                          context_->tensor_options()
                                              .dtype(output_dtype_)
                                              .device_index(
                                                  context_->device()));
        }

        const bool use_mt256_native_contract_capacity =
            input_dtype_ == ScalarType::BFloat16 &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", false);
        const int sorted_padding_per_expert =
            use_mt256_native_contract_capacity ? 256 : 128;
        const int64_t sorted_capacity =
            static_cast<int64_t>(capacity_full_m_) * topk_ +
            static_cast<int64_t>(num_experts_) * sorted_padding_per_expert;
        const int64_t routed_rows = static_cast<int64_t>(capacity_full_m_) * topk_;
        const int64_t aggregate_max_tiles = (sorted_capacity + 127) / 128;
        const int64_t max_tiles =
            (routed_rows + 127) / 128 +
            static_cast<int64_t>(num_experts_) * world_size_;
        auto int_options = context_->tensor_options()
                               .dtype(ScalarType::Int)
                               .device_index(context_->device());
        sorted_token_ids_ = empty({sorted_capacity}, int_options);
        sorted_gather_index_ = empty({sorted_capacity}, int_options);
        expert_rank_counts_ =
            empty({num_experts_ * world_size_}, int_options);
        expert_rank_counters_ =
            empty({num_experts_ * world_size_}, int_options);
        expert_rank_offsets_ =
            empty({num_experts_ * world_size_ + 1}, int_options);
        global_expert_offsets_ =
            empty({num_experts_ + 1}, int_options);
        rank_cursors_ =
            empty({num_experts_ * (world_size_ + 1)}, int_options);
        rank_tile_info_ = empty({max_tiles, 5}, int_options);
        rank_tile_problem_info_ = empty({max_tiles, 8}, int_options);
        tile_experts_ = empty({max_tiles}, int_options);
        tile_starts_ = empty({max_tiles}, int_options);
        tile_rank_starts_ = empty({max_tiles}, int_options);
        tile_rank_ends_ = empty({max_tiles}, int_options);
        rank_tile_ready_flags_ = empty({max_tiles}, int_options);
        ntiles_ = empty({1}, int_options);
        n_chunk_done_groups_ =
            (output_dim_ + group_n_chunk_size_ - 1) / group_n_chunk_size_;
        n_chunk_done_counts_ =
            empty({n_chunk_done_groups_}, int_options);
        n_chunk_done_flags_ =
            empty({n_chunk_done_groups_}, int_options);
        CUDA_CHECK(hipMemset(n_chunk_done_counts_.data_ptr(),
                             0,
                             sizeof(int) * n_chunk_done_groups_));
        CUDA_CHECK(hipMemset(n_chunk_done_flags_.data_ptr(),
                             0,
                             sizeof(int) * n_chunk_done_groups_));
        CUDA_CHECK(hipMemset(rank_tile_ready_flags_.data_ptr(),
                             0,
                             sizeof(int) * max_tiles));
        sorted_capacity_ = static_cast<int32_t>(sorted_capacity);
        aggregate_max_tiles_ = static_cast<int32_t>(aggregate_max_tiles);
        max_tiles_ = static_cast<int32_t>(max_tiles);
        barrier_buffers_ = create_tensor_list(
            {signal_capacity_}, ScalarType::Int, context_, true);
        barrier_buffer_ = barrier_buffers_[local_rank_];
        barrier_buffer_.zero_();
        CUDA_CHECK(hipMemset(barrier_buffer_.data_ptr(),
                             0,
                             sizeof(int) * signal_capacity_));
        signal_one_ = ones({1}, int_options);
        signal_value_tensor_ = empty({1}, int_options);

#ifndef FTRAIN_SHM_USE_NVSHMEM
        sync_buffers_ = create_tensor_list(
            {world_size_}, ScalarType::Int, context_, true);
        sync_buffers_[rank_].zero_();
#endif

        // Compact Native-KN waits on a cross-rank payload rendezvous before
        // launching GEMM.  A highest-priority comm stream lets that fence
        // finish while the compute stream builds metadata, hiding most of
        // the otherwise exposed barrier latency.  Keep an explicit env value
        // authoritative and leave all other routes at their historical
        // default.
        const char *asm_variant_env = std::getenv(
            "FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT");
        const std::string asm_variant = asm_variant_env != nullptr
            ? std::string(asm_variant_env)
            : std::string();
        const bool compact_variant =
            asm_variant == "v36_nn_compact_direct" ||
            asm_variant ==
                "v36_dynamic_nn_ailk_compact_direct_output_nchunk_signal";
        const bool compact_route_requested = compact_variant ||
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_ROWS", false);
        comm_stream_high_priority_ = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_HIGH_PRIORITY_COMM",
            compact_route_requested);
        if (comm_stream_high_priority_)
        {
            int least_priority = 0;
            int greatest_priority = 0;
            CUDA_CHECK(hipDeviceGetStreamPriorityRange(&least_priority,
                                                       &greatest_priority));
            CUDA_CHECK(hipStreamCreateWithPriority(&comm_stream_,
                                                   hipStreamNonBlocking,
                                                   greatest_priority));
        }
        else
        {
            CUDA_CHECK(
                hipStreamCreateWithFlags(&comm_stream_, hipStreamNonBlocking));
        }
        int pack_least_priority = 0;
        int pack_greatest_priority = 0;
        CUDA_CHECK(hipDeviceGetStreamPriorityRange(&pack_least_priority,
                                                   &pack_greatest_priority));
        weight_pack_stream_.initialize();
        CUDA_CHECK(hipStreamCreateWithPriority(&wgrad_stream_,
                                               hipStreamNonBlocking,
                                               pack_least_priority));
        CUDA_CHECK(hipEventCreateWithFlags(&weight_pack_start_event_,
                                           hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&weight_pack_done_event_,
                                           hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&wgrad_metadata_ready_event_,
                                           hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&wgrad_done_event_,
                                           hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&wgrad_input_ready_event_,
                                           hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&ready_event_, hipEventDisableTiming));
        if (route_spec_.nk_new_a2a_chunk || enk_ring_chunk_ready_)
            CUDA_CHECK(hipEventCreateWithFlags(&a2a_consumer_done_event_, hipEventDisableTiming));
        CUDA_CHECK(
            hipEventCreateWithFlags(&remote_ready_event_, hipEventDisableTiming));
        CUDA_CHECK(hipEventCreateWithFlags(&chunk_bootstrap_event_,
                                           hipEventDisableTiming));
        push_streams_.resize(world_size_, nullptr);
        push_done_events_.resize(world_size_, nullptr);
        int push_stream_priority = get_int_from_env_local(
            "FTRAIN_AG_GROUP_GEMM_V11_PUSH_STREAM_PRIORITY", 0);
        int least_priority = 0;
        int greatest_priority = 0;
        CUDA_CHECK(hipDeviceGetStreamPriorityRange(&least_priority,
                                                   &greatest_priority));
        push_stream_priority = std::max(
            greatest_priority,
            std::min(push_stream_priority, least_priority));
        for (int dst_rank = 0; dst_rank < world_size_; ++dst_rank)
        {
            if (dst_rank == rank_)
            {
                continue;
            }
            CUDA_CHECK(hipStreamCreateWithPriority(&push_streams_[dst_rank],
                                                   hipStreamNonBlocking,
                                                   push_stream_priority));
            CUDA_CHECK(hipEventCreateWithFlags(&push_done_events_[dst_rank],
                                               hipEventDisableTiming));
        }
    }

    ~AGGroupGemmImpl()
    {
        if (a2a_submit_thread_.joinable())
            a2a_submit_thread_.join();
        if (a2a_consumer_done_event_ != nullptr)
        {
            hipEventSynchronize(a2a_consumer_done_event_);
            hipEventDestroy(a2a_consumer_done_event_);
            a2a_consumer_done_event_ = nullptr;
        }
        if (d_input_ptrs_ != nullptr)
        {
            hipFree(d_input_ptrs_);
            d_input_ptrs_ = nullptr;
        }
        if (d_rank_m_offsets_ != nullptr)
        {
            hipFree(d_rank_m_offsets_);
            d_rank_m_offsets_ = nullptr;
        }
        if (ready_event_ != nullptr)
        {
            hipEventDestroy(ready_event_);
            ready_event_ = nullptr;
        }
        if (remote_ready_event_ != nullptr)
        {
            hipEventDestroy(remote_ready_event_);
            remote_ready_event_ = nullptr;
        }
        if (chunk_bootstrap_event_ != nullptr)
        {
            hipEventDestroy(chunk_bootstrap_event_);
            chunk_bootstrap_event_ = nullptr;
        }
        if (wgrad_stream_ != nullptr)
        {
            hipStreamDestroy(wgrad_stream_);
            wgrad_stream_ = nullptr;
        }
        if (weight_pack_start_event_ != nullptr)
        {
            hipEventDestroy(weight_pack_start_event_);
            weight_pack_start_event_ = nullptr;
        }
        if (weight_pack_done_event_ != nullptr)
        {
            hipEventDestroy(weight_pack_done_event_);
            weight_pack_done_event_ = nullptr;
        }
        if (wgrad_metadata_ready_event_ != nullptr)
        {
            hipEventDestroy(wgrad_metadata_ready_event_);
            wgrad_metadata_ready_event_ = nullptr;
        }
        if (wgrad_done_event_ != nullptr)
        {
            hipEventDestroy(wgrad_done_event_);
            wgrad_done_event_ = nullptr;
        }
        if (wgrad_input_ready_event_ != nullptr)
        {
            hipEventDestroy(wgrad_input_ready_event_);
            wgrad_input_ready_event_ = nullptr;
        }
        if (comm_stream_ != nullptr)
        {
            hipStreamDestroy(comm_stream_);
            comm_stream_ = nullptr;
        }
        for (auto event : push_done_events_)
        {
            if (event != nullptr)
            {
                hipEventDestroy(event);
            }
        }
        for (auto stream : push_streams_)
        {
            if (stream != nullptr)
            {
                hipStreamDestroy(stream);
            }
        }
    }

    int next_signal_value()
    {
        if (signal_value_ == std::numeric_limits<int>::max())
        {
            signal_value_ = 0;
        }
        return ++signal_value_;
    }

    bool supports_ragged_chunk_a2a() const
    {
        return route_spec_.nk_new_a2a_chunk && route_spec_.nk_32k;
    }

    bool uses_ring_reuse_credit() const
    {
        return (route_spec_.kind == AGRouteKind::NativeKn || enk_ring_chunk_ready_) &&
            world_size_ > 1;
    }

    void prepare_ring_reuse(int signal_value, hipStream_t stream)
    {
        if (!uses_ring_reuse_credit())
        {
            return;
        }
        const int credit_slot = signal_capacity_ - 1;
        const int sender_rank = (rank_ + 1) % world_size_;
        const int receiver_rank = (rank_ - 1 + world_size_) % world_size_;
        CUDA_CHECK(hipStreamWriteValue32(
            stream,
            barrier_buffers_[sender_rank].data_ptr<int>() + credit_slot,
            static_cast<uint32_t>(signal_value),
            0));
        for (int channel = 0; channel < ring_channels_; ++channel)
        {
            hipStream_t channel_stream = channel == 0
                ? push_streams_[receiver_rank]
                : ring_extra_streams_[receiver_rank][channel - 1];
            CUDA_CHECK(hipStreamWaitValue32(
                channel_stream,
                barrier_buffer_.data_ptr<int>() + credit_slot,
                static_cast<uint32_t>(signal_value),
                hipStreamWaitValueEq));
        }
    }

    void join_a2a_submitter()
    {
        if (a2a_submit_thread_.joinable())
            a2a_submit_thread_.join();
        if (a2a_submit_error_)
            std::rethrow_exception(std::exchange(a2a_submit_error_, nullptr));
    }

    void launch_chunk_a2a(const Tensor &input, int signal_value, hipStream_t compute_stream)
    {
        constexpr int prefix = 2;
        constexpr int chunks = 8;
        constexpr int inflight = 4;
        const int chunk_rows = (local_m_ + chunks - 1) / chunks;
        const size_t row_bytes = static_cast<size_t>(reduction_dim_) * input.element_size();
        const size_t owner_offset = static_cast<size_t>(local_m_offset_) * row_bytes;
        int *completion = barrier_buffer_.data_ptr<int>() + 128;
        auto submit = [this, input, signal_value, chunk_rows, row_bytes, owner_offset,
                       completion, destinations = std::vector<int>{}]
            (int first_chunk, int final_chunk, bool finish,
             const std::vector<hipStream_t> &transfer_streams) mutable {
            for (int chunk = first_chunk; chunk < final_chunk; ++chunk)
            {
                for (int step = 1; step < world_size_; ++step)
                {
                    const int distance = (step + 1) / 2;
                    const int destination = (rank_ + ((step & 1) ? -distance : distance) +
                                             world_size_) % world_size_;
                    hipStream_t push_stream = transfer_streams[destination];
                    CUDA_CHECK(hipStreamWaitEvent(push_stream, ready_event_, 0));
                    if (destinations.size() >= inflight)
                    {
                        const int prior = destinations[destinations.size() - inflight];
                        CUDA_CHECK(hipStreamWaitEvent(push_stream, push_done_events_[prior], 0));
                    }
                    const int first_row = chunk * chunk_rows;
                    const int rows = std::max(0, std::min(chunk_rows, local_m_ - first_row));
                    const size_t offset = static_cast<size_t>(first_row) * row_bytes;
                    if (rows > 0)
                    {
                        CUDA_CHECK(hipMemcpyPeerAsync(
                            static_cast<char *>(input_buffers_[destination].data_ptr()) +
                                owner_offset + offset, destination,
                            static_cast<const char *>(input.data_ptr()) + offset, rank_,
                            static_cast<size_t>(rows) * row_bytes, push_stream));
                    }
                    int *signals = barrier_buffers_[destination].data_ptr<int>();
                    CUDA_CHECK(hipStreamWriteValue32(push_stream,
                        signals + 64 + rank_ * chunks + chunk,
                        static_cast<uint32_t>(signal_value), 0));
                    if (chunk == chunks - 1)
                        CUDA_CHECK(hipStreamWriteValue32(push_stream, signals + rank_,
                            static_cast<uint32_t>(signal_value), 0));
                    CUDA_CHECK(hipEventRecord(push_done_events_[destination], push_stream));
                    destinations.push_back(destination);
                }
            }
            if (finish)
            {
                hipStream_t finish_stream = transfer_streams[(rank_ - 1 + world_size_) % world_size_];
                for (int destination = 0; destination < world_size_; ++destination)
                {
                    if (destination != rank_)
                        CUDA_CHECK(hipStreamWaitEvent(finish_stream, push_done_events_[destination], 0));
                }
                CUDA_CHECK(hipStreamWriteValue32(finish_stream, completion,
                    static_cast<uint32_t>(signal_value), 0));
            }
        };
        submit(0, prefix, false, push_streams_);
        auto transfer_streams = push_streams_;
        using GetHardwareQueue = hipError_t (*)(hipStream_t, uint64_t *);
        static auto get_hardware_queue = reinterpret_cast<GetHardwareQueue>(
            dlsym(RTLD_DEFAULT, "duStreamGetHwQueue"));
        if (get_hardware_queue != nullptr)
        {
            CUDA_CHECK(hipEventRecord(chunk_bootstrap_event_, comm_stream_));
            uint64_t compute_queue = 0;
            uint64_t completion_queue = 0;
            CUDA_CHECK(get_hardware_queue(compute_stream, &compute_queue));
            CUDA_CHECK(get_hardware_queue(comm_stream_, &completion_queue));
            std::vector<uint64_t> push_queues(world_size_, 0);
            std::vector<hipStream_t> independent_streams;
            for (int peer = 0; peer < world_size_; ++peer)
            {
                if (peer == rank_)
                    continue;
                CUDA_CHECK(get_hardware_queue(push_streams_[peer], &push_queues[peer]));
                if (push_queues[peer] != compute_queue &&
                    push_queues[peer] != completion_queue)
                    independent_streams.push_back(push_streams_[peer]);
            }
            FTRAIN_REQUIRE(!independent_streams.empty(), route_,
                        " requires a communication hardware queue independent of "
                        "the compute and completion-wait queues");
            size_t replacement = 0;
            for (int peer = 0; peer < world_size_; ++peer)
            {
                if (peer != rank_ && (push_queues[peer] == compute_queue ||
                                     push_queues[peer] == completion_queue))
                {
                    transfer_streams[peer] = independent_streams[
                        replacement++ % independent_streams.size()];
                    if (rank_ == 0 && get_bool_from_env(
                            "FTRAIN_AG_GROUP_GEMM_DEBUG_ROUTE", false))
                        printf("[moe] A2A queue reroute peer=%d blocked_queue=%lu "
                               "compute_queue=%lu completion_queue=%lu\n",
                               peer, push_queues[peer], compute_queue, completion_queue);
                }
            }
        }
        a2a_deferred_submit_ = [this, submit = std::move(submit), completion, signal_value,
                                transfer_streams = std::move(transfer_streams)]() mutable {
            CUDA_CHECK(hipStreamWaitValue32(comm_stream_, completion,
                static_cast<uint32_t>(signal_value), hipStreamWaitValueEq));
            CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
            auto suffix = [this, submit, transfer_streams = std::move(transfer_streams),
                           device = local_rank_]() mutable {
                try
                {
                    CUDA_CHECK(hipSetDevice(device));
                    submit(prefix, chunks, true, transfer_streams);
                }
                catch (...)
                {
                    a2a_submit_error_ = std::current_exception();
                }
            };
            try
            {
                a2a_submit_thread_ = std::thread(suffix);
            }
            catch (const std::system_error &)
            {
                suffix();
                join_a2a_submitter();
            }
        };
    }

    int launch_rank_signal_allgather(const Tensor &input,
                                     hipStream_t stream,
                                     bool use_push_copy,
                                     bool use_copy_nocu,
                                     bool use_chunk_signal,
                                     int signal_chunk_rows,
                                     bool signal_local_chunks_early,
                                     bool v11_chunk_bootstrap,
                                     bool native_kn_weight,
                                     bool native_kn_chunk,
                                     bool v36_rank_chunk_signal)
    {
        if (get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_BENCHMARK_PRELOADED_INPUT", false))
        {
            if (get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_PRELOADED_INPUT_SIGNALS", false))
            {
                return 1;
            }
            CUDA_CHECK(hipEventRecord(remote_ready_event_, stream));
            return 0;
        }

        const size_t row_bytes =
            static_cast<size_t>(reduction_dim_) * input.element_size();
        const size_t local_bytes = input.numel() * input.element_size();
        const size_t local_offset_bytes =
            static_cast<size_t>(local_m_offset_) * row_bytes;
        void *src = input.data_ptr();
        auto *local_input_base = static_cast<char *>(input_buffer_.data_ptr());
        auto *local_signal_base = static_cast<int *>(barrier_buffer_.data_ptr());
        // Ragged raw/internal/body namespaces are parameterized so world sizes
        // below 8 work.  The assembly stage map and consumer acquire must use
        // the same body slot base.
        const int ragged_body_slot_base = get_int_from_env_local(
            "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_BODY_SLOT_BASE", 24);
        // Opt-in owner-rank ring transport for bodies (such as public
        // v36_direct) that poll whole owner-rank signals themselves.  This is
        // intentionally independent from V40_TRUE_RING so the v36 dynamic
        // scheduler remains active.
        const bool push_ring_gate = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_PUSH_RING_GATE", false);
        const bool enk_ring_owner_signals = route_spec_.nk_new_a2a_ring;
        const bool enk_ring_peer_signals = enk_ring_owner_signals &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_NK_NEW_RING_PEER_SIGNALS", false);
        if (route_spec_.kn_ring_chunk_ready)
        {
            FTRAIN_CHECK(use_chunk_signal && signal_chunk_rows > 0 && ring_channels_ == 1);
            const int max_rows = *std::max_element(rank_m_sizes_.begin(), rank_m_sizes_.end());
            const int chunks = (max_rows + signal_chunk_rows - 1) / signal_chunk_rows;
            const int total_chunks = world_size_ * chunks;
            constexpr int raw_base = 32;
            constexpr int body_base = 128;
            FTRAIN_REQUIRE(total_chunks <= 64 &&
                            body_base + total_chunks < signal_capacity_,
                        route_, " requires at most 64 physical chunks; increase "
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_RING_CHUNK_ROWS");
            FTRAIN_CHECK(!kn_chunk_transport_);
            const int signal_value = next_signal_value();
            CUDA_CHECK(hipMemcpyAsync(signal_value_tensor_.data_ptr(),
                                      &signal_value_, sizeof(int),
                                      hipMemcpyHostToDevice, stream));
            prepare_ring_reuse(signal_value, stream);
            CUDA_CHECK(hipEventRecord(ready_event_, stream));
            CUDA_CHECK(hipMemcpyAsync(local_input_base + local_offset_bytes,
                                      src, local_bytes, hipMemcpyDeviceToDevice, stream));
            ag_copy_remote_ragged_ring_light_acquire_group_impl(
                local_signal_base, static_cast<uint32_t>(signal_value),
                body_base + rank_ * chunks, chunks, -1, 0, stream);
            kn_chunk_transport_ = [=]() {
                const int dst_rank = (rank_ - 1 + world_size_) % world_size_;
                hipStream_t ring_stream = push_streams_[dst_rank];
                auto *dst_input = static_cast<char *>(input_buffers_[dst_rank].data_ptr());
                auto *dst_signals = static_cast<int *>(barrier_buffers_[dst_rank].data_ptr());
                CUDA_CHECK(hipStreamWaitEvent(ring_stream, ready_event_, 0));
                CUDA_CHECK(hipStreamWaitEvent(comm_stream_, ready_event_, 0));
                for (int chunk = 0; chunk < chunks; ++chunk)
                {
                    const int row_start = chunk * signal_chunk_rows;
                    for (int stage = 1; stage < world_size_; ++stage)
                    {
                        const int owner = (rank_ + stage - 1) % world_size_;
                        const int chunk_id = owner * chunks + chunk;
                        if (stage > 1)
                            CUDA_CHECK(hipStreamWaitValue32(
                                ring_stream, local_signal_base + raw_base + chunk_id,
                                static_cast<uint32_t>(signal_value), hipStreamWaitValueGte));
                        const int rows = std::max(0, std::min(
                            signal_chunk_rows, rank_m_sizes_[owner] - row_start));
                        const size_t chunk_offset = static_cast<size_t>(row_start) * row_bytes;
                        const size_t owner_offset =
                            static_cast<size_t>(rank_m_offsets_[owner]) * row_bytes;
                        if (rows > 0)
                            CUDA_CHECK(hipMemcpyPeerAsync(
                                dst_input + owner_offset + chunk_offset, dst_rank,
                                stage == 1
                                    ? static_cast<const char *>(src) + chunk_offset
                                    : local_input_base + owner_offset + chunk_offset,
                                rank_, static_cast<size_t>(rows) * row_bytes,
                                ring_stream));
                        CUDA_CHECK(hipMemcpyPeerAsync(
                            dst_signals + raw_base + chunk_id, dst_rank,
                            signal_value_tensor_.data_ptr(), rank_, sizeof(int), ring_stream));
                    }
                    for (int stage = 1; stage < world_size_; ++stage)
                    {
                        const int owner = (rank_ + stage) % world_size_;
                        const int raw_slot = raw_base + owner * chunks + chunk;
                        CUDA_CHECK(hipStreamWaitValue32(
                            comm_stream_, local_signal_base + raw_slot,
                            static_cast<uint32_t>(signal_value), hipStreamWaitValueGte));
                        ag_copy_remote_ragged_ring_light_acquire_group_impl(
                            local_signal_base, static_cast<uint32_t>(signal_value),
                            raw_slot, 1, -1, body_base - raw_base, comm_stream_);
                    }
                }
                CUDA_CHECK(hipEventRecord(push_done_events_[dst_rank], ring_stream));
                CUDA_CHECK(hipStreamWaitEvent(comm_stream_, push_done_events_[dst_rank], 0));
                CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
            };
            return signal_value;
        }
        if (use_push_copy || v36_rank_chunk_signal || push_ring_gate)
        {
            const hipMemcpyKind copy_kind =
                use_copy_nocu ? hipMemcpyDeviceToDeviceNoCU
                              : hipMemcpyDeviceToDevice;
            const bool raw_kn_v36_overlap = native_kn_weight &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_RAW_V36_OVERLAP", false);
            // Native-KN uses destination-owned peer copies because its remote
            // rows are read directly by hand-written MFMA kernels. Variant 19
            // restores the v36 scheduling/order for the raw FTrain ABI, while
            // retaining this gfx936 visibility requirement. A cross-device
            // hipMemcpyAsync(DeviceToDevice) can expose stale peer data even
            // after the signal/event chain; hipMemcpyPeerAsync establishes the
            // correct source/destination device ownership.  Keep an explicit
            // override for A/B measurements, but make the safe API the Native-
            // KN default so correctness does not depend on an environment
            // knob.
            const bool use_peer_api = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_PUSH_PEER_API",
                native_kn_weight);
            // Experimental Native-KN publication path.  The payload already
            // uses hipMemcpyPeerAsync into the destination GPU.  Publish its
            // rank-ready value with a stream-ordered remote write on that
            // same push stream, rather than a second generic D2D copy.  This
            // preserves payload-before-signal ordering when remote pushes
            // overlap the local copy.  Keep the historical signal copy as
            // the default until correctness has been established.
            const bool compact_kn_route = native_kn_weight &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_ROWS", false);
            const bool native_kn_stream_release_signal =
                native_kn_weight && (use_peer_api || raw_kn_v36_overlap) &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_STREAM_RELEASE_SIGNAL",
                    raw_kn_v36_overlap);
            const bool native_kn_payload_system_fence =
                raw_kn_v36_overlap && use_peer_api &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_PAYLOAD_SYSTEM_FENCE", false);
            const bool v36_stage_ordered_requested =
                get_bool_from_env_with_legacy_local(
                    "FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE",
                    "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_STAGE_ORDERED",
                    false);
            const int v36_n_tiles = (output_dim_ + 255) / 256;
            // The ring transport is independent of the output-N tile count;
            // keep the existing V36 dynamic scheduler bound separate from
            // ring transport and validate the assembly launch directly.
            const bool v40_ring_env_requested = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_V40_TRUE_RING", false);
            const bool v40_true_ring_requested = v36_n_tiles > 0 &&
                v40_ring_env_requested;
            const bool v36_dynamic = v36_stage_ordered_requested &&
                !v40_true_ring_requested &&
                v36_n_tiles > 0 && v36_n_tiles <= 64 &&
                get_bool_from_env_with_legacy_local(
                    "FTRAIN_AG_GROUP_GEMM_RANK_TILE_DYNAMIC",
                    "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_DYNAMIC",
                    true);
            const bool v36_stage_ordered =
                v36_stage_ordered_requested &&
                (v36_n_tiles == 8 || v36_dynamic ||
                 v40_true_ring_requested);
            const bool v36_v5_fallback =
                v36_stage_ordered_requested && !v36_stage_ordered;
            const bool v40_true_ring = v36_stage_ordered && !v36_dynamic &&
                v40_true_ring_requested;
            const bool v40_partial_ready = v40_true_ring && get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY", false);
            // Ragged fixed-ring keeps the fixed-M seven-stage forwarding
            // sequence, but split signal namespaces: low slots are physical
            // owner readiness consumed by variant 17, while high slots are
            // internal ring arrivals used only by forwarding.
            const bool ragged_fixed_owner_signals = ragged_m_ &&
                native_kn_weight && v40_true_ring && get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_RING_OWNER_SIGNALS",
                    false);
            const bool ragged_fixed_chunk_acquire =
                ragged_fixed_owner_signals &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_CHUNK_ACQUIRE", false);
            FTRAIN_CHECK(!ragged_fixed_owner_signals || ring_channels_ == 1)
                << "Ragged fixed-ring currently requires the single-channel "
                   "fixed-M V40 transport";
            const bool v40_partial_ready_chunk_ring = v40_true_ring &&
                ((native_kn_chunk && get_bool_from_env(
                      "FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_CHUNKS", false)) ||
                 (v36_rank_chunk_signal && v40_partial_ready &&
                  !ragged_fixed_chunk_acquire));
            const bool v36_rank_chunk_ring =
                v36_rank_chunk_signal && !ragged_fixed_chunk_acquire;
            const bool v12_route =
                v36_stage_ordered ||
                (get_bool_from_env(
                     "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V12_READY_QUEUE", false) &&
                 v36_n_tiles == 8);
            const bool force_physical_signal_layout = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_FORCE_PHYSICAL_SIGNAL_LAYOUT", false);
            const bool use_stage_signal_layout = !force_physical_signal_layout &&
                !v36_v5_fallback &&
                (v36_stage_ordered ||
                 get_bool_from_env(
                     "FTRAIN_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT", false));
            const bool benchmark_early_stage_signals =
                v36_stage_ordered && !v40_true_ring && get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_BENCHMARK_EARLY_STAGE_SIGNALS",
                    false);
            // Native-KN and v12-style routes use the historical early remote
            // push by default. The explicit environment override remains
            // authoritative for correctness/performance A/B tests.
            const bool start_remote_before_local = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_PUSH_EARLY_REMOTE",
                v12_route);
            if (rank_ == 0 &&
                get_bool_from_env("FTRAIN_AG_GROUP_GEMM_DEBUG_ROUTE", false))
            {
                printf("[moe] AG push ordering: native_kn=%d raw_kn_v36=%d "
                       "v12_route=%d peer_api=%d early_remote=%d "
                       "partial_ready=%d\\n",
                       native_kn_weight ? 1 : 0,
                       raw_kn_v36_overlap ? 1 : 0,
                       v12_route ? 1 : 0,
                       use_peer_api ? 1 : 0,
                       start_remote_before_local ? 1 : 0,
                       v40_partial_ready ? 1 : 0);
            }
            const int signal_value = next_signal_value();
            CUDA_CHECK(hipMemcpyAsync(signal_value_tensor_.data_ptr(),
                                      &signal_value_,
                                      sizeof(int),
                                      hipMemcpyHostToDevice,
                                      stream));
            if (world_size_ == 1)
            {
                // Single-rank ETP has no remote owner and therefore no push
                // stream/event handles.  Local input and body-facing slots
                // were published above; mark remote ready and return.
                CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
                return signal_value;
            }
            if (v36_rank_chunk_ring)
            {
                CUDA_CHECK(hipMemsetAsync(local_signal_base,
                                          0,
                                          sizeof(int) * (signal_capacity_ -
                                              (uses_ring_reuse_credit() ? 1 : 0)),
                                          stream));
            }
            prepare_ring_reuse(signal_value, stream);
            if (route_spec_.nk_new_a2a_chunk && signal_value > 1)
            {
                for (int peer = 0; peer < world_size_; ++peer)
                {
                    if (peer != rank_)
                        CUDA_CHECK(hipStreamWriteValue32(stream,
                            barrier_buffers_[peer].data_ptr<int>() + 129 + rank_,
                            static_cast<uint32_t>(signal_value - 1), 0));
                }
                for (int peer = 0; peer < world_size_; ++peer)
                {
                    if (peer != rank_)
                        CUDA_CHECK(hipStreamWaitValue32(stream, local_signal_base + 129 + peer,
                            static_cast<uint32_t>(signal_value - 1), hipStreamWaitValueGte));
                }
            }
            if (start_remote_before_local)
            {
                CUDA_CHECK(hipEventRecord(ready_event_, stream));
            }

            const int local_signal_index = ragged_fixed_owner_signals
                ? rank_
                : (use_stage_signal_layout ? 0 : rank_);
            if (benchmark_early_stage_signals)
            {
                // Benchmark-only: input_buffer_ must already contain the same
                // immutable input from a fully synchronized bootstrap call.
                CUDA_CHECK(hipMemcpyAsync(
                    local_signal_base + local_signal_index,
                    signal_value_tensor_.data_ptr(),
                    sizeof(int),
                    copy_kind,
                    stream));
            }
            CUDA_CHECK(hipMemcpyAsync(local_input_base + local_offset_bytes,
                                      src,
                                      local_bytes,
                                      copy_kind,
                                      stream));
            if (ragged_fixed_owner_signals)
            {
                ag_group_gemm_system_fence_impl(stream);
                ag_group_gemm_publish_signal_release_impl(
                    local_signal_base + rank_,
                    static_cast<uint32_t>(signal_value),
                    stream);
                if (!ragged_fixed_chunk_acquire)
                {
                    ag_group_gemm_publish_signal_release_impl(
                        local_signal_base + ragged_body_slot_base + rank_,
                        static_cast<uint32_t>(signal_value),
                        stream);
                }
                else
                {
                    const int chunk_slot_base = get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_CHUNK_SLOT_BASE", 32);
                    const int max_rank_rows = *std::max_element(
                        rank_m_sizes_.begin(), rank_m_sizes_.end());
                    const int chunks_per_rank =
                        (max_rank_rows + signal_chunk_rows - 1) /
                        signal_chunk_rows;
                    const int local_chunks = (rank_m_sizes_[rank_] +
                                              signal_chunk_rows - 1) /
                                             signal_chunk_rows;
                    for (int chunk = 0; chunk < local_chunks &&
                                       chunk < chunks_per_rank; ++chunk)
                    {
                        ag_group_gemm_publish_signal_release_impl(
                            local_signal_base + chunk_slot_base + chunk,
                            static_cast<uint32_t>(signal_value),
                            stream);
                    }
                }
            }
            else if (!benchmark_early_stage_signals && !v36_rank_chunk_signal)
            {
                CUDA_CHECK(hipMemcpyAsync(
                    local_signal_base + local_signal_index,
                    signal_value_tensor_.data_ptr(),
                    sizeof(int),
                    copy_kind,
                    stream));
            }
            else if (v36_rank_chunk_ring)
            {
                ag_signal_local_chunks_impl(1024,
                                            world_size_,
                                            rank_,
                                            full_m_,
                                            signal_chunk_rows,
                                            signal_value,
                                            local_signal_base,
                                            stream);
            }

            if (!start_remote_before_local)
            {
                CUDA_CHECK(hipEventRecord(ready_event_, stream));
            }

            if (route_spec_.nk_new_a2a_chunk)
            {
                FTRAIN_CHECK((supports_ragged_chunk_a2a() || !ragged_m_) &&
                           local_m_ >= (supports_ragged_chunk_a2a() ? 1 : 8) && world_size_ == 8 &&
                           use_peer_api && !use_chunk_signal && !v36_rank_chunk_signal &&
                           !v40_true_ring && !push_ring_gate);
                launch_chunk_a2a(input, signal_value, stream);
                return signal_value;
            }
            if (v36_rank_chunk_ring)
            {
                const int dst_rank =
                    (rank_ - 1 + world_size_) % world_size_;
                hipStream_t ring_stream = push_streams_[dst_rank];
                auto *dst_input_base = static_cast<char *>(
                    input_buffers_[dst_rank].data_ptr());
                auto *dst_signal_base = static_cast<int *>(
                    barrier_buffers_[dst_rank].data_ptr());
                const int local_rows = full_m_ / world_size_;
                const int chunks_per_rank =
                    (local_rows + signal_chunk_rows - 1) / signal_chunk_rows;
                CUDA_CHECK(hipStreamWaitEvent(ring_stream, ready_event_, 0));
                for (int stage = 1; stage < world_size_; ++stage)
                {
                    const int owner_rank =
                        (rank_ + stage - 1) % world_size_;
                    if (stage > 1)
                    {
                        const int prior_chunk = owner_rank * chunks_per_rank;
                        CUDA_CHECK(hipStreamWaitValue32(
                            ring_stream,
                            local_signal_base + prior_chunk,
                            static_cast<uint32_t>(signal_value),
                            hipStreamWaitValueGte));
                    }
                    const void *stage_src = stage == 1
                        ? src
                        : local_input_base + owner_rank * local_bytes;
                    CUDA_CHECK(hipMemcpyPeerAsync(
                        dst_input_base + owner_rank * local_bytes,
                        dst_rank,
                        stage_src,
                        rank_,
                        local_bytes,
                        ring_stream));
                    for (int rank_chunk = 0;
                         rank_chunk < chunks_per_rank;
                         ++rank_chunk)
                    {
                        ag_group_gemm_publish_signal_release_impl(
                            dst_signal_base +
                                owner_rank * chunks_per_rank + rank_chunk,
                            static_cast<uint32_t>(signal_value),
                            ring_stream);
                    }
                }
                CUDA_CHECK(hipEventRecord(push_done_events_[dst_rank],
                                          ring_stream));
                CUDA_CHECK(hipStreamWaitEvent(
                    comm_stream_, push_done_events_[dst_rank], 0));
                CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
                return signal_value;
            }

            if (v40_partial_ready_chunk_ring)
            {
                // Keep the V40 ring transport, but publish a small number of
                // chunk signals instead of one signal for the whole rank
                // segment.  The chunk-capable V36 scheduler can then start a
                // tile as soon as its input interval is available without
                // turning every 256-row tile into a separate DMA launch.
                const int dst_rank =
                    (rank_ - 1 + world_size_) % world_size_;
                hipStream_t ring_stream = push_streams_[dst_rank];
                auto *dst_input_base = static_cast<char *>(
                    input_buffers_[dst_rank].data_ptr());
                auto *dst_signal_base = static_cast<int *>(
                    barrier_buffers_[dst_rank].data_ptr());
                const int local_rows = full_m_ / world_size_;
                const int chunks_per_rank =
                    (local_rows + signal_chunk_rows - 1) / signal_chunk_rows;
                if (v36_rank_chunk_signal)
                {
                    ag_signal_local_chunks_impl(1024,
                                                world_size_,
                                                v36_rank_chunk_signal ? 0 : rank_,
                                                full_m_,
                                                signal_chunk_rows,
                                                signal_value,
                                                local_signal_base,
                                                stream);
                }
                CUDA_CHECK(hipStreamWaitEvent(ring_stream, ready_event_, 0));
                for (int stage = 1; stage < world_size_; ++stage)
                {
                    const int owner_rank =
                        (rank_ + stage - 1) % world_size_;
                    for (int rank_chunk = 0;
                         rank_chunk < chunks_per_rank;
                         ++rank_chunk)
                    {
                        if (stage > 1)
                        {
                            const int prior_stage = v36_rank_chunk_signal
                                ? (owner_rank - rank_ + world_size_) % world_size_
                                : owner_rank;
                            const int prior_chunk =
                                prior_stage * chunks_per_rank + rank_chunk;
                            CUDA_CHECK(hipStreamWaitValue32(
                                ring_stream,
                                local_signal_base + prior_chunk,
                                static_cast<uint32_t>(signal_value),
                                hipStreamWaitValueGte));
                        }
                        const int local_row_start =
                            rank_chunk * signal_chunk_rows;
                        const int rows_in_chunk = std::min(
                            signal_chunk_rows, local_rows - local_row_start);
                        if (rows_in_chunk <= 0)
                        {
                            continue;
                        }
                        const size_t src_byte_offset =
                            static_cast<size_t>(local_row_start) *
                            reduction_dim_ * input.element_size();
                        const size_t dst_byte_offset =
                            static_cast<size_t>(owner_rank * local_rows +
                                                local_row_start) *
                            reduction_dim_ * input.element_size();
                        const size_t chunk_bytes =
                            static_cast<size_t>(rows_in_chunk) *
                            reduction_dim_ * input.element_size();
                        const void *stage_src = stage == 1
                            ? static_cast<const char *>(src) +
                                  src_byte_offset
                            : local_input_base + dst_byte_offset;
                        CUDA_CHECK(hipMemcpyPeerAsync(
                            dst_input_base + dst_byte_offset,
                            dst_rank,
                            stage_src,
                            rank_,
                            chunk_bytes,
                            ring_stream));
                        const int signal_stage = v36_rank_chunk_signal
                            ? (owner_rank - dst_rank + world_size_) % world_size_
                            : owner_rank;
                        const int chunk_id =
                            signal_stage * chunks_per_rank + rank_chunk;
                        if (v36_rank_chunk_signal)
                        {
                            ag_group_gemm_publish_signal_release_impl(
                                dst_signal_base + chunk_id,
                                static_cast<uint32_t>(signal_value),
                                ring_stream);
                        }
                        else
                        {
                            CUDA_CHECK(hipStreamWriteValue32(
                                ring_stream,
                                dst_signal_base + chunk_id,
                                static_cast<uint32_t>(signal_value),
                                0));
                        }
                    }
                }
                CUDA_CHECK(hipEventRecord(push_done_events_[dst_rank],
                                          ring_stream));
                CUDA_CHECK(hipStreamWaitEvent(
                    comm_stream_, push_done_events_[dst_rank], 0));
                CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
                return signal_value;
            }

            if (enk_ring_chunk_ready_ && ragged_m_)
            {
                FTRAIN_CHECK(world_size_ == 8 && ring_channels_ == 1);
                FTRAIN_REQUIRE(!get_bool_from_env("FTRAIN_ENK_RING_EARLY_SUBMIT", false) &&
                    !get_bool_from_env("FTRAIN_ENK_RING_HISTORY_SCHEDULE", false),
                    "Dynamic ENK currently requires non-early physical chunk mode");
                FTRAIN_CHECK(local_m_ > 0 && signal_capacity_ > 128);
                FTRAIN_CHECK(!enk_chunk_transport_);
                FTRAIN_CHECK(!start_remote_before_local ||
                    !get_bool_from_env("FTRAIN_ENK_RING_EARLY_SUBMIT", false));
                const int chunks = 8;
                const int max_rows = *std::max_element(rank_m_sizes_.begin(), rank_m_sizes_.end());
                const int chunk_rows = (max_rows + chunks - 1) / chunks;
                const int destination = (rank_ - 1 + world_size_) % world_size_;
                hipStream_t ring_stream = push_streams_[destination];
                auto *destination_input = static_cast<char *>(input_buffers_[destination].data_ptr());
                auto *destination_signals = static_cast<int *>(barrier_buffers_[destination].data_ptr());
                enk_chunk_transport_ = [=]() {
                    CUDA_CHECK(hipStreamWaitEvent(ring_stream, ready_event_, 0));
                    for (int chunk = 0; chunk < chunks; ++chunk)
                    {
                        const int first_row = chunk * chunk_rows;
                        const size_t chunk_offset = static_cast<size_t>(first_row) * row_bytes;
                        for (int stage = 1; stage < world_size_; ++stage)
                        {
                            const int owner = (rank_ + stage - 1) % world_size_;
                            const int rows = std::max(0, std::min(chunk_rows, rank_m_sizes_[owner] - first_row));
                            const int slot = 64 + owner * chunks + chunk;
                            if (stage > 1)
                                CUDA_CHECK(hipStreamWaitValue32(ring_stream, local_signal_base + slot,
                                    static_cast<uint32_t>(signal_value), hipStreamWaitValueGte));
                            const size_t owner_offset = static_cast<size_t>(rank_m_offsets_[owner]) * row_bytes;
                            const void *payload = stage == 1
                                ? static_cast<const char *>(src) + chunk_offset
                                : local_input_base + owner_offset + chunk_offset;
                            if (rows > 0)
                                CUDA_CHECK(hipMemcpyPeerAsync(destination_input + owner_offset + chunk_offset,
                                    destination, payload, rank_, static_cast<size_t>(rows) * row_bytes,
                                    ring_stream));
                            CUDA_CHECK(hipStreamWriteValue32(ring_stream, destination_signals + slot,
                                static_cast<uint32_t>(signal_value), 0));
                            if (chunk == chunks - 1)
                            {
                                CUDA_CHECK(hipStreamWriteValue32(ring_stream,
                                    destination_signals + world_size_ + stage,
                                    static_cast<uint32_t>(signal_value), 0));
                                CUDA_CHECK(hipStreamWriteValue32(ring_stream,
                                    destination_signals + (use_stage_signal_layout ? stage : owner),
                                    static_cast<uint32_t>(signal_value), 0));
                            }
                        }
                    }
                    CUDA_CHECK(hipEventRecord(push_done_events_[destination], ring_stream));
                    CUDA_CHECK(hipStreamWaitEvent(comm_stream_, push_done_events_[destination], 0));
                    CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
                };
                if (!get_bool_from_env("FTRAIN_ENK_RING_EARLY_SUBMIT", false))
                {
                    auto submit = std::move(enk_chunk_transport_);
                    enk_chunk_transport_ = nullptr;
                    submit();
                }
                return signal_value;
            }

            if (enk_ring_chunk_ready_)
            {
                FTRAIN_CHECK(world_size_ == 8 && !ragged_m_ && ring_channels_ == 1);
                FTRAIN_CHECK(local_m_ >= 8 && signal_capacity_ > 128);
                FTRAIN_CHECK(!enk_chunk_transport_);
                FTRAIN_CHECK(!start_remote_before_local ||
                    !get_bool_from_env("FTRAIN_ENK_RING_EARLY_SUBMIT", false));
                const int chunks = 8;
                const int chunk_rows = (local_m_ + chunks - 1) / chunks;
                const int destination = (rank_ - 1 + world_size_) % world_size_;
                hipStream_t ring_stream = push_streams_[destination];
                auto *destination_input = static_cast<char *>(input_buffers_[destination].data_ptr());
                auto *destination_signals = static_cast<int *>(barrier_buffers_[destination].data_ptr());
                enk_chunk_transport_ = [=]() {
                    CUDA_CHECK(hipStreamWaitEvent(ring_stream, ready_event_, 0));
                    for (int chunk = 0; chunk < chunks; ++chunk)
                    {
                        const int first_row = chunk * chunk_rows;
                        const int rows = std::max(0, std::min(chunk_rows, local_m_ - first_row));
                        const size_t chunk_offset = static_cast<size_t>(first_row) * row_bytes;
                        for (int stage = 1; stage < world_size_; ++stage)
                        {
                            const int owner = (rank_ + stage - 1) % world_size_;
                            const int slot = 64 + owner * chunks + chunk;
                            if (stage > 1)
                                CUDA_CHECK(hipStreamWaitValue32(ring_stream, local_signal_base + slot,
                                    static_cast<uint32_t>(signal_value), hipStreamWaitValueGte));
                            const size_t owner_offset = static_cast<size_t>(owner) * local_bytes;
                            const void *payload = stage == 1
                                ? static_cast<const char *>(src) + chunk_offset
                                : local_input_base + owner_offset + chunk_offset;
                            if (rows > 0)
                                CUDA_CHECK(hipMemcpyPeerAsync(destination_input + owner_offset + chunk_offset,
                                    destination, payload, rank_, static_cast<size_t>(rows) * row_bytes,
                                    ring_stream));
                            CUDA_CHECK(hipStreamWriteValue32(ring_stream, destination_signals + slot,
                                static_cast<uint32_t>(signal_value), 0));
                            if (chunk == chunks - 1)
                            {
                                CUDA_CHECK(hipStreamWriteValue32(ring_stream,
                                    destination_signals + world_size_ + stage,
                                    static_cast<uint32_t>(signal_value), 0));
                                CUDA_CHECK(hipStreamWriteValue32(ring_stream,
                                    destination_signals + (use_stage_signal_layout ? stage : owner),
                                    static_cast<uint32_t>(signal_value), 0));
                            }
                        }
                    }
                    CUDA_CHECK(hipEventRecord(push_done_events_[destination], ring_stream));
                    CUDA_CHECK(hipStreamWaitEvent(comm_stream_, push_done_events_[destination], 0));
                    CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
                };
                if (!get_bool_from_env("FTRAIN_ENK_RING_EARLY_SUBMIT", false))
                {
                    auto submit = std::move(enk_chunk_transport_);
                    enk_chunk_transport_ = nullptr;
                    submit();
                }
                return signal_value;
            }

            if (v40_true_ring || push_ring_gate || enk_ring_owner_signals)
            {
                // Push one segment per stage to the previous rank. From stage
                // two onward, wait until this rank has received the segment
                // that it must forward. Signal slots are ring arrival stages.
                // Channel count > 1 stripes each stage across independent
                // high-priority streams; stage arrival remains a single
                // release after every channel completes.
                const int dst_rank =
                    (rank_ - 1 + world_size_) % world_size_;
                if (enk_ring_owner_signals && get_bool_from_env(
                        "FTRAIN_AG_GROUP_GEMM_DEBUG_ROUTE", false))
                {
                    int push_priority = 0;
                    CUDA_CHECK(hipStreamGetPriority(
                        push_streams_[dst_rank], &push_priority));
                    FTRAIN_LOG_INFO("AG ENK owner ring rank=%d dst=%d stages=%d "
                                  "owner_slots=%s arrival_slot_base=%d "
                                  "push_priority=%d publish=%s",
                                  rank_, dst_rank, world_size_ - 1,
                                  use_stage_signal_layout ? "stage" : "physical",
                                  world_size_, push_priority,
                                  enk_ring_peer_signals ? "peer_copy" : "stream_write");
                }
                auto *dst_input_base =
                    static_cast<char *>(input_buffers_[dst_rank].data_ptr());
                auto *dst_signal_base =
                    static_cast<int *>(barrier_buffers_[dst_rank].data_ptr());
                auto channel_stream = [&](int channel) -> hipStream_t {
                    return channel == 0 ? push_streams_[dst_rank]
                                        : ring_extra_streams_[dst_rank][channel - 1];
                };
                // Push-ring-gate reserves the low slots for whole owner-rank
                // readiness consumed by the public v36 body, and uses higher
                // slots for internal ring-forwarding arrival.
                auto ring_arrival_slot = [&](int stage) {
                    return (push_ring_gate || ragged_fixed_owner_signals ||
                            enk_ring_owner_signals)
                        ? world_size_ + stage
                        : stage;
                };
                auto publish_ring_signal = [&](hipStream_t publication_stream,
                                                int slot) {
                    if (enk_ring_peer_signals)
                    {
                        CUDA_CHECK(hipMemcpyPeerAsync(
                            dst_signal_base + slot,
                            dst_rank,
                            signal_value_tensor_.data_ptr(),
                            rank_,
                            sizeof(int),
                            publication_stream));
                    }
                    else
                    {
                        CUDA_CHECK(hipStreamWriteValue32(
                            publication_stream,
                            dst_signal_base + slot,
                            static_cast<uint32_t>(signal_value),
                            0));
                    }
                };
                if (ring_channels_ == 1)
                {
                    hipStream_t ring_stream = channel_stream(0);
                    CUDA_CHECK(hipStreamWaitEvent(ring_stream, ready_event_, 0));
                    for (int stage = 1; stage < world_size_; ++stage)
                    {
                        const int owner_rank =
                            (rank_ + stage - 1) % world_size_;
                        const size_t owner_bytes = ragged_m_
                            ? static_cast<size_t>(
                                  rank_m_sizes_[owner_rank]) * row_bytes
                            : local_bytes;
                        const size_t owner_offset = ragged_m_
                            ? static_cast<size_t>(
                                  rank_m_offsets_[owner_rank]) * row_bytes
                            : static_cast<size_t>(owner_rank) * local_bytes;
                        if (stage > 1)
                        {
                            CUDA_CHECK(hipStreamWaitValue32(
                                ring_stream,
                                local_signal_base + ring_arrival_slot(stage - 1),
                                static_cast<uint32_t>(signal_value),
                                hipStreamWaitValueGte));
                        }
                        const void *stage_src = stage == 1
                            ? src
                            : local_input_base + owner_offset;
                        if (use_peer_api)
                        {
                            CUDA_CHECK(hipMemcpyPeerAsync(
                                dst_input_base + owner_offset,
                                dst_rank,
                                stage_src,
                                rank_,
                                owner_bytes,
                                ring_stream));
                        }
                        else
                        {
                            CUDA_CHECK(hipMemcpyAsync(
                                dst_input_base + owner_offset,
                                stage_src,
                                owner_bytes,
                                copy_kind,
                                ring_stream));
                        }
                        if (ragged_fixed_owner_signals)
                        {
                            if (get_bool_from_env(
                                    "FTRAIN_AG_GROUP_GEMM_RAGGED_PAYLOAD_READBACK",
                                    false))
                            {
                                CUDA_CHECK(hipMemcpyPeerAsync(
                                    signal_one_.data_ptr(),
                                    rank_,
                                    dst_input_base + owner_offset,
                                    dst_rank,
                                    sizeof(int),
                                    ring_stream));
                            }
                            CUDA_CHECK(hipMemcpyPeerAsync(
                                dst_signal_base + ring_arrival_slot(stage),
                                dst_rank,
                                signal_value_tensor_.data_ptr(),
                                rank_,
                                sizeof(int),
                                ring_stream));
                            if (get_bool_from_env(
                                    "FTRAIN_AG_GROUP_GEMM_RAGGED_PRODUCER_RELEASE",
                                    false))
                            {
                                ag_group_gemm_system_fence_impl(ring_stream);
                                ag_group_gemm_publish_signal_release_impl(
                                    dst_signal_base + owner_rank,
                                    static_cast<uint32_t>(signal_value),
                                    ring_stream);
                            }
                            else
                            {
                                CUDA_CHECK(hipMemcpyPeerAsync(
                                    dst_signal_base + owner_rank,
                                    dst_rank,
                                    signal_value_tensor_.data_ptr(),
                                    rank_,
                                    sizeof(int),
                                    ring_stream));
                            }
                        }
                        else
                        {
                            publish_ring_signal(
                                ring_stream, ring_arrival_slot(stage));
                        }
                        if (!ragged_fixed_owner_signals &&
                            (push_ring_gate || enk_ring_owner_signals))
                        {
                            publish_ring_signal(
                                ring_stream,
                                enk_ring_owner_signals && use_stage_signal_layout
                                    ? stage : owner_rank);
                        }
                    }
                    CUDA_CHECK(hipEventRecord(push_done_events_[dst_rank],
                                              ring_stream));
                }
                else
                {
                    for (int channel = 0; channel < ring_channels_; ++channel)
                    {
                        CUDA_CHECK(hipStreamWaitEvent(channel_stream(channel),
                                                      ready_event_, 0));
                    }
                    for (int stage = 1; stage < world_size_; ++stage)
                    {
                        const int owner_rank =
                            (rank_ + stage - 1) % world_size_;
                        if (stage > 1)
                        {
                                            for (int channel = 0; channel < ring_channels_; ++channel)
                                            {
                                                CUDA_CHECK(hipStreamWaitValue32(
                                                    channel_stream(channel),
                                                    local_signal_base + ring_arrival_slot(stage - 1),
                                    static_cast<uint32_t>(signal_value),
                                    hipStreamWaitValueGte));
                            }
                        }
                        const void *stage_src = stage == 1
                            ? src
                            : local_input_base + owner_rank * local_bytes;
                        const size_t slice_bytes = (local_bytes +
                            ring_channels_ - 1) / ring_channels_;
                        for (int channel = 0; channel < ring_channels_; ++channel)
                        {
                            const size_t offset = std::min(
                                local_bytes,
                                static_cast<size_t>(channel) * slice_bytes);
                            const size_t bytes = std::min(slice_bytes,
                                local_bytes - offset);
                            if (bytes == 0)
                            {
                                continue;
                            }
                            if (use_peer_api)
                            {
                                CUDA_CHECK(hipMemcpyPeerAsync(
                                    dst_input_base + owner_rank * local_bytes +
                                        offset,
                                    dst_rank,
                                    static_cast<const char *>(stage_src) + offset,
                                    rank_,
                                    bytes,
                                    channel_stream(channel)));
                            }
                            else
                            {
                                CUDA_CHECK(hipMemcpyAsync(
                                    dst_input_base + owner_rank * local_bytes +
                                        offset,
                                    static_cast<const char *>(stage_src) + offset,
                                    bytes,
                                    copy_kind,
                                    channel_stream(channel)));
                            }
                            CUDA_CHECK(hipEventRecord(
                                ring_channel_done_events_[dst_rank]
                                    [(stage - 1) * ring_channels_ + channel],
                                channel_stream(channel)));
                        }
                        for (int channel = 0; channel < ring_channels_; ++channel)
                        {
                            CUDA_CHECK(hipStreamWaitEvent(
                                comm_stream_,
                                ring_channel_done_events_[dst_rank]
                                    [(stage - 1) * ring_channels_ + channel],
                                0));
                        }
                        publish_ring_signal(
                            comm_stream_, ring_arrival_slot(stage));
                        if (push_ring_gate || enk_ring_owner_signals)
                        {
                            publish_ring_signal(
                                comm_stream_,
                                enk_ring_owner_signals && use_stage_signal_layout
                                    ? stage : owner_rank);
                        }
                    }
                    CUDA_CHECK(hipEventRecord(push_done_events_[dst_rank],
                                              comm_stream_));
                }
                CUDA_CHECK(hipStreamWaitEvent(
                    comm_stream_, push_done_events_[dst_rank], 0));
                if (ragged_fixed_owner_signals && get_bool_from_env(
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE", true))
                {
                    // Consumer-side acquire.  The producer-side payload signal
                    // only says that its copy engine touched the destination;
                    // variant 17 can still observe stale cached lines on this
                    // device.  Wait for each owner's raw signal, read that
                    // owner's local payload with system-scope atomics, and only
                    // then publish the body-facing acquired slot.
                    int acquire_grid = get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_GRID",
                        16);
                    acquire_grid = std::max(1, std::min(acquire_grid, 256));
                    int acquire_stride_words = get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_STRIDE_WORDS",
                        32);
                    acquire_stride_words = std::max(
                        1, std::min(acquire_stride_words, 1024));
                    int acquire_probe_mode = get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_MODE",
                        3);
                    acquire_probe_mode =
                        std::max(0, std::min(acquire_probe_mode, 3));
                    int acquire_group_size = get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_CONSUMER_ACQUIRE_GROUP_SIZE",
                        1);
                    acquire_group_size = std::max(
                        1,
                        std::min(acquire_group_size,
                                  std::min(world_size_, 2)));
                    const bool grouped_light_acquire =
                        !ragged_fixed_chunk_acquire &&
                        acquire_probe_mode == 3;
                    if (grouped_light_acquire)
                    {
                        for (int owner_begin = 0; owner_begin < world_size_;
                             owner_begin += acquire_group_size)
                        {
                            const int owner_end = std::min(
                                world_size_, owner_begin + acquire_group_size);
                            int stream_owner = owner_begin;
                            if (stream_owner == rank_ && owner_end > owner_begin)
                            {
                                stream_owner = owner_begin + 1;
                            }
                            if (stream_owner >= owner_end)
                            {
                                continue;
                            }
                            hipStream_t acquire_stream =
                                push_streams_[stream_owner];
                            for (int owner_rank = owner_begin;
                                 owner_rank < owner_end;
                                 ++owner_rank)
                            {
                                if (owner_rank == rank_)
                                {
                                    continue;
                                }
                                CUDA_CHECK(hipStreamWaitValue32(
                                    acquire_stream,
                                    local_signal_base + owner_rank,
                                    static_cast<uint32_t>(signal_value),
                                    hipStreamWaitValueGte));
                            }
                            ag_copy_remote_ragged_ring_light_acquire_group_impl(
                                local_signal_base,
                                static_cast<uint32_t>(signal_value),
                                owner_begin,
                                owner_end - owner_begin,
                                rank_,
                                ragged_body_slot_base,
                                acquire_stream);
                        }
                    }
                    for (int owner_rank = 0;
                         !grouped_light_acquire &&
                         owner_rank < world_size_;
                         ++owner_rank)
                    {
                        if (owner_rank == rank_)
                        {
                            continue;
                        }
                        hipStream_t acquire_stream =
                            push_streams_[owner_rank];
                        const bool debug_acquire = get_bool_from_env(
                            "FTRAIN_AG_GROUP_GEMM_DEBUG_RAGGED_ACQUIRE", false);
                        if (debug_acquire)
                        {
                            fprintf(stderr,
                                    "[ragged-acquire] rank=%d owner=%d "
                                    "enqueue\n",
                                    rank_,
                                    owner_rank);
                        }
                        CUDA_CHECK(hipStreamWaitValue32(
                            acquire_stream,
                            local_signal_base + owner_rank,
                            static_cast<uint32_t>(signal_value),
                            hipStreamWaitValueGte));
                        const size_t owner_offset = static_cast<size_t>(
                            rank_m_offsets_[owner_rank]) * row_bytes;
                        if (!ragged_fixed_chunk_acquire)
                        {
                            const size_t owner_bytes = static_cast<size_t>(
                                rank_m_sizes_[owner_rank]) * row_bytes;
                            ag_copy_remote_ragged_ring_payload_acquire_impl(
                                acquire_grid,
                                1024,
                                local_input_base + owner_offset,
                                owner_bytes,
                                local_signal_base + 3 * world_size_ +
                                    owner_rank,
                                local_signal_base + ragged_body_slot_base +
                                    owner_rank,
                            local_signal_base + 4 * world_size_ +
                                owner_rank,
                            static_cast<uint32_t>(signal_value),
                            acquire_stride_words,
                            acquire_probe_mode,
                            acquire_stream);
                            if (debug_acquire)
                            {
                                CUDA_CHECK(hipStreamSynchronize(acquire_stream));
                                int observed = -1;
                                CUDA_CHECK(hipMemcpy(
                                    &observed,
                                    local_signal_base + 3 * world_size_ +
                                        owner_rank,
                                    sizeof(int),
                                    hipMemcpyDeviceToHost));
                                fprintf(stderr,
                                        "[ragged-acquire] rank=%d owner=%d "
                                        "signal=%d\n",
                                        rank_,
                                        owner_rank,
                                        observed);
                            }
                            continue;
                        }
                        const int max_rank_rows = *std::max_element(
                            rank_m_sizes_.begin(), rank_m_sizes_.end());
                        const int chunks_per_rank =
                            (max_rank_rows + signal_chunk_rows - 1) /
                            signal_chunk_rows;
                        const int chunk_slot_base = get_int_from_env_local(
                            "FTRAIN_AG_GROUP_GEMM_RAGGED_CHUNK_SLOT_BASE", 32);
                        const int counter_slot_base = get_int_from_env_local(
                            "FTRAIN_AG_GROUP_GEMM_RAGGED_CHUNK_COUNTER_BASE",
                            128);
                        const int owner_chunks = (rank_m_sizes_[owner_rank] +
                                                  signal_chunk_rows - 1) /
                                                 signal_chunk_rows;
                        const int body_stage =
                            (owner_rank - rank_ + world_size_) % world_size_;
                        for (int chunk = 0;
                             chunk < owner_chunks && chunk < chunks_per_rank;
                             ++chunk)
                        {
                            const size_t chunk_bytes = std::min(
                                static_cast<size_t>(signal_chunk_rows) *
                                    row_bytes,
                                static_cast<size_t>(
                                    rank_m_sizes_[owner_rank]) * row_bytes -
                                    static_cast<size_t>(chunk) *
                                        signal_chunk_rows * row_bytes);
                            const int acquired_slot =
                                chunk_slot_base +
                                body_stage * chunks_per_rank + chunk;
                            ag_copy_remote_ragged_ring_payload_acquire_impl(
                                acquire_grid,
                                1024,
                                local_input_base + owner_offset +
                                    static_cast<size_t>(chunk) *
                                        signal_chunk_rows * row_bytes,
                                chunk_bytes,
                                local_signal_base + acquired_slot,
                                local_signal_base + acquired_slot,
                                local_signal_base + counter_slot_base +
                                    owner_rank * chunks_per_rank + chunk,
                                static_cast<uint32_t>(signal_value),
                                acquire_stride_words,
                                acquire_probe_mode,
                                acquire_stream);
                            if (debug_acquire)
                            {
                                CUDA_CHECK(hipStreamSynchronize(acquire_stream));
                                int observed = -1;
                                CUDA_CHECK(hipMemcpy(
                                    &observed,
                                    local_signal_base + acquired_slot,
                                    sizeof(int),
                                    hipMemcpyDeviceToHost));
                                fprintf(stderr,
                                        "[ragged-acquire] rank=%d owner=%d "
                                        "chunk=%d slot=%d signal=%d\n",
                                        rank_,
                                        owner_rank,
                                        chunk,
                                        acquired_slot,
                                        observed);
                            }
                        }
                    }
                }
                CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
                return signal_value;
            }

            // Keep the V5 default (all peers in flight), while the target
            // V12 ready-queue route uses two rank-adjacent waves by default.
            // Explicit knobs and the old V11 knob remain available.
            int push_max_inflight = world_size_ - 1;
            const bool v11_route = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V11", false);
            const char *push_cap_env = std::getenv(
                "FTRAIN_AG_GROUP_GEMM_PUSH_MAX_INFLIGHT");
            if (v36_dynamic)
            {
                // At the 8K raw-KN shape, a second in-flight peer copy
                // competes with the 12-wave TN-loader body on gfx936. One
                // active push reduces that contention. The 16K shape still
                // benefits from two-way copy concurrency, so keep this
                // override deliberately narrow.
                const bool raw_kn_tnloader_small = raw_kn_v36_overlap &&
                    get_bool_from_env(
                        "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_V19_TNLOADER_BODY",
                        false) &&
                    full_m_ <= 8192 && reduction_dim_ <= 8192 &&
                    output_dim_ <= 1024;
                const int dynamic_default_inflight =
                    raw_kn_tnloader_small
                        ? 1
                        : (v36_n_tiles <= 2 ? world_size_ - 1 : 2);
                push_max_inflight = get_int_from_env_with_legacy_local(
                    "FTRAIN_AG_GROUP_GEMM_RANK_TILE_PUSH_MAX_INFLIGHT",
                    "FTRAIN_AG_GROUP_GEMM_V36_DYNAMIC_PUSH_MAX_INFLIGHT",
                    dynamic_default_inflight);
            }
            else if (v36_stage_ordered)
            {
                push_max_inflight = 2;
            }
            else if (push_cap_env != nullptr && *push_cap_env != '\0')
            {
                push_max_inflight = get_int_from_env_local(
                    "FTRAIN_AG_GROUP_GEMM_PUSH_MAX_INFLIGHT",
                    push_max_inflight);
            }
            else if (v11_route)
            {
                push_max_inflight = get_int_from_env_local(
                    "FTRAIN_AG_GROUP_GEMM_V11_PUSH_MAX_INFLIGHT",
                    push_max_inflight);
            }
            else if (v12_route)
            {
                push_max_inflight = native_kn_weight ? 4 : 2;
            }
            // Permit an explicit cap to override the route's conservative
            // default.  This is especially important for Megatron FC1 KN:
            // each rank contributes six expanded copies, so the seven peer
            // transfers are much larger than the raw v36 input transfer.
            // An explicit cap still overrides the route default for A/B
            // measurements and deployment-specific tuning.
            if (push_cap_env != nullptr && *push_cap_env != '\0')
            {
                push_max_inflight = get_int_from_env_local(
                    "FTRAIN_AG_GROUP_GEMM_PUSH_MAX_INFLIGHT",
                    push_max_inflight);
            }
            push_max_inflight = std::max(
                1, std::min(push_max_inflight, world_size_ - 1));
            const bool zigzag_order = v36_stage_ordered ||
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_PUSH_ZIGZAG_ORDER", v12_route);
            const auto dst_rank_for_step = [&](int step) {
                if (!zigzag_order)
                {
                    return (rank_ - step + world_size_) % world_size_;
                }
                const int distance = (step + 1) / 2;
                const int delta = (step & 1) ? -distance : distance;
                return (rank_ + delta + world_size_) % world_size_;
            };
            const auto signal_index_for_dst = [&](int dst_rank) {
                if (!use_stage_signal_layout)
                {
                    return rank_;
                }
                const int forward =
                    (rank_ - dst_rank + world_size_) % world_size_;
                if (!zigzag_order || forward == 0)
                {
                    return forward;
                }
                const int backward = world_size_ - forward;
                return forward <= backward ? 2 * forward - 1
                                           : 2 * backward;
            };
            for (int step = 1; step < world_size_; ++step)
            {
                const int dst_rank = dst_rank_for_step(step);
                hipStream_t push_stream = push_streams_[dst_rank];
                auto *dst_input_base =
                    static_cast<char *>(input_buffers_[dst_rank].data_ptr());
                auto *dst_signal_base =
                    static_cast<int *>(barrier_buffers_[dst_rank].data_ptr());
                if (benchmark_early_stage_signals)
                {
                    CUDA_CHECK(
                        hipStreamWaitEvent(push_stream, ready_event_, 0));
                    CUDA_CHECK(hipMemcpyAsync(
                        dst_signal_base + signal_index_for_dst(dst_rank),
                        signal_value_tensor_.data_ptr(),
                        sizeof(int),
                        copy_kind,
                        push_stream));
                }
                if (step > push_max_inflight)
                {
                    const int prior_step = step - push_max_inflight;
                    const int prior_dst = dst_rank_for_step(prior_step);
                    CUDA_CHECK(hipStreamWaitEvent(
                        push_stream, push_done_events_[prior_dst], 0));
                }
                if (!benchmark_early_stage_signals)
                {
                    CUDA_CHECK(
                        hipStreamWaitEvent(push_stream, ready_event_, 0));
                }
                if (use_peer_api)
                {
                    CUDA_CHECK(hipMemcpyPeerAsync(
                        dst_input_base + local_offset_bytes,
                        dst_rank,
                        src,
                        rank_,
                        local_bytes,
                        push_stream));
                }
                else
                {
                    CUDA_CHECK(hipMemcpyAsync(
                        dst_input_base + local_offset_bytes,
                        src,
                        local_bytes,
                        copy_kind,
                        push_stream));
                }
                if (!benchmark_early_stage_signals)
                {
                    if (native_kn_payload_system_fence)
                    {
                        ag_group_gemm_system_fence_impl(push_stream);
                    }
                    if (native_kn_stream_release_signal)
                    {
                        // A host-enqueued stream write is not a system-scope
                        // release on gfx936 IPC mappings.  Launch a tiny
                        // device-side release after the peer DMA instead;
                        // the launch is ordered on this push stream, so the
                        // remote consumer cannot observe the epoch early.
                        ag_group_gemm_publish_signal_release_impl(
                            dst_signal_base + signal_index_for_dst(dst_rank),
                            static_cast<uint32_t>(signal_value),
                            push_stream);
                    }
                    else if (native_kn_weight && use_peer_api &&
                             get_bool_from_env(
                                 "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_PEER_SIGNAL_COPY",
                                 true))
                    {
                        // Keep payload and publication on the same peer-copy
                        // engine and stream.  A generic DeviceToDevice copy to
                        // an IPC pointer does not provide the same destination
                        // ownership/visibility contract as the preceding
                        // hipMemcpyPeerAsync payload on gfx936.
                        CUDA_CHECK(hipMemcpyPeerAsync(
                            dst_signal_base + signal_index_for_dst(dst_rank),
                            dst_rank,
                            signal_value_tensor_.data_ptr(),
                            rank_,
                            sizeof(int),
                            push_stream));
                    }
                    else
                    {
                        CUDA_CHECK(hipMemcpyAsync(
                            dst_signal_base + signal_index_for_dst(dst_rank),
                            signal_value_tensor_.data_ptr(),
                            sizeof(int),
                            copy_kind,
                            push_stream));
                    }
                }
                CUDA_CHECK(
                    hipEventRecord(push_done_events_[dst_rank], push_stream));
            }

            for (int dst_rank = 0; dst_rank < world_size_; ++dst_rank)
            {
                if (dst_rank != rank_)
                {
                    CUDA_CHECK(hipStreamWaitEvent(
                        comm_stream_, push_done_events_[dst_rank], 0));
                }
            }
            // push_done_events_ only cover this rank's outgoing copies.  A
            // consumer-side wait on remote_ready_event_ therefore does not,
            // by itself, prove that every peer has finished writing this
            // rank's input buffer.  Compact Native-KN needs a real cross-rank
            // rendezvous after all outgoing DMA operations; only then does
            // remote_ready_event_ describe a fully published all-gather
            // buffer on every rank.
            if (native_kn_weight &&
                (get_bool_from_env(
                     "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_WAIT_REMOTE_READY", false) ||
                 compact_kn_route))
            {
#ifndef FTRAIN_SHM_USE_NVSHMEM
                all_ranks_sync(comm_stream_, sync_buffers_, rank_);
#else
                all_ranks_sync(comm_stream_, std::nullopt, std::nullopt);
#endif
            }
            CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
            return signal_value;
        }

        const bool use_chunk_push = use_chunk_signal &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_PUSH", false);
        // Native-KN chunk ABI consumes the same per-rank chunk slots as the
        // chunk-copy path.  Enable that path directly for the dedicated
        // Megatron FC1 route; otherwise only the local rank is signalled and
        // remote chunk payloads never reach the destination buffers.
        // Tile-ready KN uses the existing seven-field chunk ABI, but moves
        // publication into the system-scope custom all-gather kernel.  Set
        // chunk_rows to one full rank shard in forward(), so each owner-rank
        // payload gets one ready slot and a compact tile can wait on its
        // rank_start..rank_end interval.
        const bool native_kn_tile_ready = native_kn_chunk &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_TILE_READY", false);
        const bool native_kn_tile_ready_peer_copy = native_kn_tile_ready &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_TILE_READY_PEER_COPY", false);
        const bool native_kn_tile_ready_v36_schedule =
            native_kn_tile_ready_peer_copy &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_TILE_READY_V36_SCHEDULE",
                false);
        const bool native_kn_direct_chunk = native_kn_weight &&
            !native_kn_chunk &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", false);
        const bool native_kn_chunk_push = (native_kn_chunk ||
                                           native_kn_direct_chunk) &&
            !native_kn_tile_ready;
        if (use_chunk_push || native_kn_chunk_push ||
            native_kn_tile_ready_peer_copy)
        {
            const bool use_peer_copy = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_PUSH_PEER_API", false);
            // The original Native-KN chunk experiment enqueues one peer DMA
            // and one signal publication per chunk.  Keep that path as the
            // default for compatibility, but allow an A/B path that uses the
            // existing device-side chunk copier once per peer stream.  This
            // preserves one ready signal per chunk while removing the host
            // enqueue amplification at large expanded M.
            const bool native_kn_chunk_custom_copy =
                (native_kn_chunk || native_kn_direct_chunk) &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_CHUNK_CUSTOM_COPY", false);
            const bool native_kn_chunk_stream_release_signal =
                (native_kn_chunk || native_kn_direct_chunk) && use_peer_copy &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_STREAM_RELEASE_SIGNAL",
                    false);
            const hipMemcpyKind data_copy_kind =
                use_copy_nocu ? hipMemcpyDeviceToDeviceNoCU
                              : hipMemcpyDeviceToDevice;
            const int local_rows = full_m_ / world_size_;
            const int chunks_per_rank =
                (local_rows + signal_chunk_rows - 1) / signal_chunk_rows;
            const size_t element_bytes = input.element_size();
            const int signal_value = next_signal_value();
            CUDA_CHECK(hipMemcpyAsync(signal_value_tensor_.data_ptr(),
                                      &signal_value_,
                                      sizeof(int),
                                      hipMemcpyHostToDevice,
                                      stream));
            CUDA_CHECK(hipMemcpyAsync(local_input_base + rank_ * local_bytes,
                                      src,
                                      local_bytes,
                                      hipMemcpyDefault,
                                      stream));
            ag_signal_local_chunks_impl(1024,
                                        world_size_,
                                        rank_,
                                        full_m_,
                                        signal_chunk_rows,
                                        signal_value,
                                        local_signal_base,
                                        stream);
            CUDA_CHECK(hipEventRecord(ready_event_, stream));

            if (native_kn_chunk_custom_copy)
            {
                int chunk_copy_grid = get_int_from_env_local(
                    "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_COPY_GRID", 8);
                chunk_copy_grid = std::max(1, std::min(chunk_copy_grid, 64));
                int chunk_copy_block = get_int_from_env_local(
                    "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_COPY_BLOCK", 1024);
                chunk_copy_block = std::max(
                    64, std::min(chunk_copy_block, 1024));
                chunk_copy_block = (chunk_copy_block / 64) * 64;

                // Run one device-side copy kernel per source rank.  Each
                // kernel walks all chunks owned by that rank and publishes
                // the matching chunk slots with a system-scope release.
                // Separate streams retain peer-level concurrency while
                // avoiding one host API sequence per chunk.
                for (int stage = 1; stage < world_size_; ++stage)
                {
                    const int src_rank =
                        (rank_ + stage) % world_size_;
                    hipStream_t rank_stream = push_streams_[src_rank];
                    FTRAIN_CHECK(rank_stream != nullptr)
                        << "Native-KN custom chunk stream is missing for rank "
                        << src_rank;
                    CUDA_CHECK(
                        hipStreamWaitEvent(rank_stream, ready_event_, 0));
                    const int signal_group_rows = get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_CHUNK_SIGNAL_GROUP_ROWS", 0);
                    const int signal_group_chunks =
                        signal_group_rows > signal_chunk_rows
                            ? signal_group_rows / signal_chunk_rows
                            : 1;
                    ag_copy_remote_b16_chunks_signal_stages_grouped_impl(
                        chunk_copy_grid,
                        chunk_copy_block,
                        8,
                        world_size_,
                        rank_,
                        full_m_,
                        reduction_dim_,
                        signal_chunk_rows,
                        signal_value,
                        stage,
                        stage + 1,
                        local_signal_base,
                        d_input_ptrs_,
                        rank_stream,
                        signal_group_chunks);
                    CUDA_CHECK(hipEventRecord(
                        push_done_events_[src_rank], rank_stream));
                    CUDA_CHECK(hipStreamWaitEvent(
                        comm_stream_, push_done_events_[src_rank], 0));
                }
                CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
                return signal_value;
            }

            auto *local_src_base = static_cast<char *>(src);
            const char *push_cap_env = std::getenv(
                "FTRAIN_AG_GROUP_GEMM_PUSH_MAX_INFLIGHT");
            int push_max_inflight =
                push_cap_env != nullptr && *push_cap_env != '\0'
                    ? get_int_from_env_local(
                          "FTRAIN_AG_GROUP_GEMM_PUSH_MAX_INFLIGHT",
                          world_size_ - 1)
                    : get_int_from_env_local(
                          "FTRAIN_AG_GROUP_GEMM_V11_PUSH_MAX_INFLIGHT",
                          native_kn_tile_ready_v36_schedule ? 2
                                                            : world_size_ - 1);
            push_max_inflight = std::max(
                1, std::min(push_max_inflight, world_size_ - 1));
            const bool zigzag_order = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_PUSH_ZIGZAG_ORDER",
                native_kn_tile_ready_v36_schedule);
            const auto dst_rank_for_step = [&](int step) {
                if (!zigzag_order)
                {
                    return (rank_ - step + world_size_) % world_size_;
                }
                const int distance = (step + 1) / 2;
                const int delta = (step & 1) ? -distance : distance;
                return (rank_ + delta + world_size_) % world_size_;
            };
            for (int step = 1; step < world_size_; ++step)
            {
                const int dst_rank = dst_rank_for_step(step);
                hipStream_t push_stream = push_streams_[dst_rank];
                if (step > push_max_inflight)
                {
                    const int prior_step = step - push_max_inflight;
                    const int prior_dst = dst_rank_for_step(prior_step);
                    CUDA_CHECK(hipStreamWaitEvent(
                        push_stream, push_done_events_[prior_dst], 0));
                }
                CUDA_CHECK(hipStreamWaitEvent(push_stream, ready_event_, 0));
                auto *dst_input_base = static_cast<char *>(
                    input_buffers_[dst_rank].data_ptr());
                auto *dst_signal_base = static_cast<int *>(
                    barrier_buffers_[dst_rank].data_ptr());
                for (int rank_chunk = 0;
                     rank_chunk < chunks_per_rank;
                     ++rank_chunk)
                {
                    const int local_row_start =
                        rank_chunk * signal_chunk_rows;
                    const int rows_in_chunk = std::min(
                        signal_chunk_rows, local_rows - local_row_start);
                    const size_t src_byte_offset =
                        static_cast<size_t>(local_row_start) * reduction_dim_ *
                        element_bytes;
                    const size_t dst_byte_offset =
                        static_cast<size_t>(rank_ * local_rows +
                                            local_row_start) *
                        reduction_dim_ * element_bytes;
                    const size_t chunk_bytes =
                        static_cast<size_t>(rows_in_chunk) * reduction_dim_ *
                        element_bytes;
                    if (use_peer_copy || native_kn_chunk_push)
                    {
                        CUDA_CHECK(hipMemcpyPeerAsync(
                            dst_input_base + dst_byte_offset,
                            dst_rank,
                            local_src_base + src_byte_offset,
                            rank_,
                            chunk_bytes,
                            push_stream));
                    }
                    else
                    {
                        CUDA_CHECK(hipMemcpyAsync(
                            dst_input_base + dst_byte_offset,
                            local_src_base + src_byte_offset,
                            chunk_bytes,
                            data_copy_kind,
                            push_stream));
                    }
                    const int chunk_id =
                        rank_ * chunks_per_rank + rank_chunk;
                    if (native_kn_chunk_stream_release_signal)
                    {
                        // The DMA and the publication must be ordered on the
                        // same push stream.  A plain D2D signal copy is not a
                        // system-scope release for the peer-visible IPC
                        // mapping consumed by the hand-written KN kernel.
                        ag_group_gemm_publish_signal_release_impl(
                            dst_signal_base + chunk_id,
                            static_cast<uint32_t>(signal_value),
                            push_stream);
                    }
                    else
                    {
                        CUDA_CHECK(hipMemcpyAsync(
                            dst_signal_base + chunk_id,
                            signal_value_tensor_.data_ptr(),
                            sizeof(int),
                            hipMemcpyDeviceToDevice,
                            push_stream));
                    }
                }
                CUDA_CHECK(
                    hipEventRecord(push_done_events_[dst_rank], push_stream));
            }
            for (int dst_rank = 0; dst_rank < world_size_; ++dst_rank)
            {
                if (dst_rank != rank_)
                {
                    CUDA_CHECK(hipStreamWaitEvent(
                        comm_stream_, push_done_events_[dst_rank], 0));
                }
            }
            CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
            return signal_value;
        }

        CUDA_CHECK(hipMemsetAsync(local_signal_base,
                                  0,
                                  sizeof(int) * signal_capacity_,
                                  stream));

        CUDA_CHECK(hipMemcpyAsync(local_input_base + rank_ * local_bytes,
                                  src,
                                  local_bytes,
                                  hipMemcpyDefault,
                                  stream));
        if (!use_chunk_signal)
        {
            CUDA_CHECK(hipMemcpyAsync(local_signal_base + rank_,
                                      signal_one_.data_ptr(),
                                      sizeof(int),
                                      hipMemcpyDeviceToDevice,
                                      stream));
        }
        else if (signal_local_chunks_early)
        {
            ag_signal_local_chunks_impl(1024,
                                        world_size_,
                                        rank_,
                                        full_m_,
                                        signal_chunk_rows,
                                        1,
                                        local_signal_base,
                                        stream);
        }

        CUDA_CHECK(hipEventRecord(ready_event_, stream));
        CUDA_CHECK(hipStreamWaitEvent(comm_stream_, ready_event_, 0));

        if (!native_kn_tile_ready)
        {
#ifndef FTRAIN_SHM_USE_NVSHMEM
            all_ranks_sync(comm_stream_, sync_buffers_, rank_);
#else
            all_ranks_sync(comm_stream_, std::nullopt, std::nullopt);
#endif
        }

        if (use_chunk_signal)
        {
            int chunk_copy_grid = get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_COPY_GRID",
                native_kn_tile_ready ? 64 : 8);
            chunk_copy_grid = std::max(1, std::min(chunk_copy_grid, 64));
            int chunk_copy_block = get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_COPY_BLOCK",
                native_kn_tile_ready ? 256 : 1024);
            chunk_copy_block =
                std::max(64, std::min(chunk_copy_block, 1024));
            chunk_copy_block = (chunk_copy_block / 64) * 64;
            if (native_kn_tile_ready && world_size_ > 1)
            {
                // Tile-ready KN has one chunk per owner rank.  Launch one
                // copy kernel per remote owner on its own stream so rank
                // payloads can make progress concurrently.  Each kernel
                // publishes only its disjoint rank slot after the payload
                // copy, preserving the system-scope visibility contract.
                for (int stage = 1; stage < world_size_; ++stage)
                {
                    const int src_rank = (rank_ + stage) % world_size_;
                    hipStream_t rank_stream = push_streams_[src_rank];
                    FTRAIN_CHECK(rank_stream != nullptr)
                        << "tile-ready rank stream is missing for rank "
                        << src_rank;
                    CUDA_CHECK(
                        hipStreamWaitEvent(rank_stream, ready_event_, 0));
                    ag_copy_remote_b16_chunks_signal_stages_impl(
                        chunk_copy_grid,
                        chunk_copy_block,
                        8,
                        world_size_,
                        rank_,
                        full_m_,
                        reduction_dim_,
                        signal_chunk_rows,
                        1,
                        stage,
                        stage + 1,
                        local_signal_base,
                        d_input_ptrs_,
                        rank_stream);
                    CUDA_CHECK(hipEventRecord(push_done_events_[src_rank],
                                              rank_stream));
                    CUDA_CHECK(hipStreamWaitEvent(
                        comm_stream_, push_done_events_[src_rank], 0));
                }
            }
            else if (signal_local_chunks_early)
            {
                if (v11_chunk_bootstrap && world_size_ > 1)
                {
                    int bootstrap_stages = get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_BOOTSTRAP_STAGES", 1);
                    bootstrap_stages = std::max(
                        1, std::min(bootstrap_stages, world_size_ - 1));
                    const int bootstrap_stage_end = 1 + bootstrap_stages;
                    ag_copy_remote_b16_chunks_signal_stages_impl(
                        chunk_copy_grid,
                        chunk_copy_block,
                        8,
                        world_size_,
                        rank_,
                        full_m_,
                        reduction_dim_,
                        signal_chunk_rows,
                        1,
                        1,
                        bootstrap_stage_end,
                        local_signal_base,
                        d_input_ptrs_,
                        comm_stream_);
                    CUDA_CHECK(
                        hipEventRecord(chunk_bootstrap_event_, comm_stream_));
                    CUDA_CHECK(
                        hipStreamWaitEvent(stream, chunk_bootstrap_event_, 0));
                    if (bootstrap_stage_end < world_size_)
                    {
                        ag_copy_remote_b16_chunks_signal_stages_impl(
                            chunk_copy_grid,
                            chunk_copy_block,
                            8,
                            world_size_,
                            rank_,
                            full_m_,
                            reduction_dim_,
                            signal_chunk_rows,
                            1,
                            bootstrap_stage_end,
                            world_size_,
                            local_signal_base,
                            d_input_ptrs_,
                            comm_stream_);
                    }
                }
                else
                {
                    ag_copy_remote_b16_chunks_signal_skip_local_impl(
                        chunk_copy_grid,
                        chunk_copy_block,
                        8,
                        world_size_,
                        rank_,
                        full_m_,
                        reduction_dim_,
                        signal_chunk_rows,
                        1,
                        local_signal_base,
                        d_input_ptrs_,
                        comm_stream_);
                }
            }
            else
            {
                ag_copy_remote_b16_chunks_signal_impl(chunk_copy_grid,
                                                      chunk_copy_block,
                                                      8,
                                                      world_size_,
                                                      rank_,
                                                      full_m_,
                                                      reduction_dim_,
                                                      signal_chunk_rows,
                                                      1,
                                                      local_signal_base,
                                                      d_input_ptrs_,
                                                      comm_stream_);
            }
        }
        else
        {
            int ag_copy_grid = get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_AG_COPY_GRID", 8);
            ag_copy_grid = std::max(1, std::min(ag_copy_grid, 64));
            int ag_copy_block = get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_AG_COPY_BLOCK", 1024);
            ag_copy_block = std::max(64, std::min(ag_copy_block, 1024));
            ag_copy_block = (ag_copy_block / 64) * 64;
            ag_copy_remote_b16_impl(ag_copy_grid,
                                    ag_copy_block,
                                    8,
                                    world_size_,
                                    rank_,
                                    full_m_,
                                    reduction_dim_,
                                    d_input_ptrs_,
                                    local_signal_base,
                                    comm_stream_);
        }
        CUDA_CHECK(hipEventRecord(remote_ready_event_, comm_stream_));
        return 1;
    }

    Tensor forward(Tensor input,
                          Tensor weight,
                          Tensor full_topk_ids,
                          std::optional<Tensor> output)
    {
        last_forward_valid_ = false;
        join_a2a_submitter();
        if (route_spec_.kn_ring_chunk_ready)
        {
            hipStreamCaptureStatus capture_status;
            CUDA_CHECK(hipStreamIsCapturing(
                context_->stream(), &capture_status));
            FTRAIN_REQUIRE(capture_status == hipStreamCaptureStatusNone,
                        route_, " does not support HIP graph capture");
        }
        if (route_spec_.nk_new_a2a_chunk || enk_ring_chunk_ready_)
        {
            hipStreamCaptureStatus capture_status;
            CUDA_CHECK(hipStreamIsCapturing(
                context_->stream(), &capture_status));
            FTRAIN_REQUIRE(capture_status == hipStreamCaptureStatusNone,
                        route_, " does not support HIP graph capture");
            FTRAIN_REQUIRE(((enk_ring_chunk_ready_ || supports_ragged_chunk_a2a()) && local_m_ > 0) ||
                            (!ragged_m_ && local_m_ >= 8),
                        route_, supports_ragged_chunk_a2a()
                            ? " requires positive rank input sizes"
                            : " requires equal rank input sizes of at least 8 rows");
            CUDA_CHECK(hipStreamWaitEvent(
                context_->stream(), a2a_consumer_done_event_, 0));
        }
        ScopedAGDefaultEnvironment default_environment(
            route_spec_.kind == AGRouteKind::Auto);
        std::unique_ptr<ScopedRouteEnvironment> route_env_guard_;
        std::unique_ptr<ScopedRaggedRouteEnvironment> ragged_env_guard_;
        if (route_spec_.kind != AGRouteKind::Auto)
        {
            route_env_guard_.reset(new ScopedRouteEnvironment(route_spec_));
        }
        else if (native_kn_ragged_ring_)
        {
            ragged_env_guard_.reset(new ScopedRaggedRouteEnvironment());
        }
        CHECK_INPUT(input, input_dtype_);
        CHECK_INPUT(weight, input_dtype_);
        CHECK_INPUT_LOOSE(full_topk_ids);
        FTRAIN_CHECK(full_topk_ids.scalar_type() == ScalarType::Int ||
                   full_topk_ids.scalar_type() == ScalarType::Long)
            << "full_topk_ids must be int32 or int64";
        CHECK_2D(input, local_m_, reduction_dim_);
        const bool input_weight_nk = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_INPUT_WEIGHT_NK", false);
        // Candidate-only native-KN [E,N_local,K] storage.  This is disjoint
        // from the legacy TN INPUT_WEIGHT_NK switch and changes only the
        // public weight shape check; the ENK code object owns addressing.
        const bool native_kn_weight_nk = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_INPUT_WEIGHT_NK", false);
        if (input_weight_nk)
        {
            // Opt-in legacy-TN experiment: the caller supplies the original
            // [E,N,K] storage while the operator dimensions remain logical
            // reduction_dim=K and output_dim=N. The validated/default path
            // continues to require the public [E,K,N] ABI.
            CHECK_3D(weight, num_experts_, output_dim_, reduction_dim_);
        }
        else if (!native_kn_weight_nk)
        {
            CHECK_3D(weight, num_experts_, reduction_dim_, output_dim_);
        }
        else
        {
            CHECK_3D(weight, num_experts_, output_dim_, reduction_dim_);
        }
        CHECK_2D(full_topk_ids, full_m_, topk_);

        Tensor out;
        if (allocate_output_on_init_)
        {
            out = output_buffer_;
        }
        else
        {
            FTRAIN_CHECK(output.has_value())
                << "output must be provided when allocate_output_on_init=false";
            out = output.value();
            CHECK_INPUT(out, output_dtype_);
            FTRAIN_CHECK(out.size(0) >= full_m_ * topk_);
            FTRAIN_CHECK(out.size(1) == output_dim_);
        }

        hipStream_t stream = context_->stream();
        if (ragged_m_ && rank_m_offsets_dirty_)
        {
            // Kernel arguments capture the host offsets by value. The device
            // update is ordered before all metadata consumers on this stream.
            ag_group_gemm_update_rank_m_offsets_impl(
                d_rank_m_offsets_,
                rank_m_offsets_.data(),
                static_cast<int>(rank_m_offsets_.size()),
                stream);
            rank_m_offsets_dirty_ = false;
        }
        // The compact route's expert-major input is also a valid topk=1
        // contract for the regular v36 body.  Keep this as an opt-in A/B
        // switch: it trades Native-KN's direct weight path for v36's larger
        // 12-wave compute body and ready-queue overlap, without changing the
        // validated compact default.
        const char *variant_env = std::getenv(
            "FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT");
        const std::string variant_before_override = variant_env != nullptr
            ? std::string(variant_env)
            : std::string();
        const bool compact_variant_selected =
            variant_before_override == "v36_nn_compact_direct" ||
            variant_before_override ==
                "v36_dynamic_nn_ailk_compact_direct_output_nchunk_signal";
        const int compact_v36_min_dim = std::max(
            1,
            get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_V36_MIN_DIM", 16384));
        const int compact_v36_global_n = output_dim_ * std::max(world_size_, 1);
        const bool compact_v36_body = compact_variant_selected &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_USE_V36_BODY", false) &&
            full_m_ >= compact_v36_min_dim &&
            compact_v36_global_n >= compact_v36_min_dim &&
            reduction_dim_ >= compact_v36_min_dim;
        if (compact_v36_body)
        {
            setenv("FTRAIN_AG_GROUP_GEMM_ASM_MT256_VARIANT",
                   "v36_direct",
                   1);
        }
        const bool use_tiled =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_HIP_HCU_MMAC", true);
        const bool use_rank_signal =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_HIP_HCU_MMAC_RANK_TILE", true);
        const bool use_direct_mmac =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_DIRECT_MMAC", true);
        const bool use_v36_topk =
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false) &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_WGM8", false) &&
            !get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", false) &&
            get_bool_from_env_with_legacy_local(
                "FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE",
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_STAGE_ORDERED",
                false);
        const bool use_mt256_native_contract =
            input_dtype_ == ScalarType::BFloat16 &&
            (topk_ == 6 || use_v36_topk) &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", false);
        const bool use_native_kn_weight =
            use_mt256_native_contract &&
            ag_group_gemm_mt256_uses_native_kn_weight(full_m_, output_dim_);
        FTRAIN_CHECK(!input_weight_nk || !use_native_kn_weight)
            << "FTRAIN_AG_GROUP_GEMM_INPUT_WEIGHT_NK=1 requires a legacy-TN "
               "variant, not a native-KN variant";
        const bool use_native_kn_chunk =
            use_mt256_native_contract &&
            ag_group_gemm_mt256_uses_native_kn_chunk(full_m_, output_dim_);
        // The compact v36-style KN selector deliberately keeps aggregate
        // expert tiles so the launcher can encode one/two physical rank spans
        // in reserved bit 6.  Do not let the generic chunk/problem-list
        // default silently force strict rank tiles for this route.
        const bool compact_kn_route =
            use_native_kn_weight &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_COMPACT_ROWS",
                              false);
        const bool use_lds_mmac =
            (input_dtype_ == ScalarType::Half ||
             input_dtype_ == ScalarType::BFloat16) &&
            !use_mt256_native_contract &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_LDS_MMAC", true);
        const bool use_packed_weight =
            use_tiled && use_direct_mmac && !use_lds_mmac &&
            !use_native_kn_weight &&
            !input_weight_nk &&
            (use_mt256_native_contract ||
             get_bool_from_env("FTRAIN_AG_GROUP_GEMM_PACK_WEIGHT", false));
        const bool requested_push_copy =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_PUSH_COPY", true);
        const bool use_mmac_shmem_b =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_MMAC_SHMEM_B", true);
        const bool requested_strict_rank_tile =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_STRICT_RANK_TILE", false);
        const bool requested_split_local_rank = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_SPLIT_LOCAL_RANK_TILE", false);
        const bool isolate_local_owner = route_spec_.nk_new_a2a_ring &&
            get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_NK_NEW_RING_OWNER_TILES", 0) == 2;
        const bool use_source_dsread_n128 =
            input_dtype_ == ScalarType::BFloat16 && use_tiled &&
            use_direct_mmac && use_lds_mmac &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_SOURCE_DSREAD_N128", true);
        const bool use_v11_chunk_signal =
            use_mt256_native_contract &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_ASM_MT256_V11", false) &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V11_CHUNK_SIGNAL", false);
        const bool use_nn_direct_chunk_signal =
            use_mt256_native_contract &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", false);
        const bool use_v36_rank_chunk_signal =
            use_native_kn_weight && !compact_kn_route &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", false);
        const bool use_dsread_fragment_layout = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_DSREAD_FRAGMENT_LAYOUT",
            use_source_dsread_n128);
        const bool use_dsread_swizzle =
            use_dsread_fragment_layout &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_DSREAD_SWIZZLE", false);
        const bool use_wg8_fragment =
            use_dsread_fragment_layout &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_WG8_FRAGMENT", false);
        const bool use_wg8_m32 =
            input_dtype_ == ScalarType::BFloat16 &&
            use_lds_mmac &&
            !use_wg8_fragment &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_WG8_M32", false);
        const bool prefer_wg16_m32_n64 =
            input_dtype_ == ScalarType::BFloat16 &&
            use_lds_mmac &&
            !use_source_dsread_n128 &&
            !use_dsread_fragment_layout &&
            !use_wg8_m32 &&
            output_dim_ >= 512 &&
            reduction_dim_ >= 512;
        const bool use_wg16_m32_n64 =
            input_dtype_ == ScalarType::BFloat16 &&
            use_lds_mmac &&
            !use_dsread_fragment_layout &&
            !use_wg8_m32 &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_WG16_M32_N64",
                              prefer_wg16_m32_n64);
        const bool use_wg16_pair_b =
            use_wg16_m32_n64 &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_WG16_M32_N64_PAIRB",
                              false);
        const bool prefer_tile_ready =
            use_source_dsread_n128 && world_size_ > 1 && output_dim_ > 256 &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_SOURCE_TILE_READY", false);
        const bool use_tile_ready =
            use_rank_signal &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_TILE_READY_FLAG",
                              prefer_tile_ready);
        int signal_chunk_rows =
            get_int_from_env_local("FTRAIN_AG_GROUP_GEMM_SIGNAL_CHUNK_ROWS", 128);
        if (signal_chunk_rows <= 0)
        {
            signal_chunk_rows = 128;
        }
        const int local_m = local_m_;
        // The chunk-ready ragged ring stays available as an explicit opt-in.
        const bool ragged_ring_chunk = ragged_m_ && use_native_kn_weight &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_V40_TRUE_RING", false) &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_RAGGED_RING_CHUNK", false);
        const bool ragged_fixed_chunk_acquire = ragged_m_ &&
            use_native_kn_weight &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_V40_TRUE_RING", false) &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_CHUNK_ACQUIRE", false);
        // Ragged native-KN uses the V40 stage-slot ring scheduler by default.
        // The chunk-ready variant keeps the historical v36-dynamic body
        // scheduler it was validated with, so it stays self-consistent.
        const bool ragged_v40_scheduler = ragged_m_ && use_native_kn_weight &&
            (!ragged_ring_chunk || route_spec_.kn_ring_chunk_ready);
        int max_rank_m_rows = local_m_;
        if (ragged_ring_chunk || ragged_fixed_chunk_acquire)
        {
            for (int rows : rank_m_sizes_)
            {
                max_rank_m_rows = std::max(max_rank_m_rows,
                                           static_cast<int>(rows));
            }
            const int default_chunk_rows = route_spec_.kn_ring_chunk_ready
                ? std::max(1536, ((max_rank_m_rows + 7) / 8 + 255) / 256 * 256)
                : 2048;
            int ragged_chunk_rows = get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_RAGGED_RING_CHUNK_ROWS", default_chunk_rows);
            signal_chunk_rows = std::max(1, ragged_chunk_rows);
            if (ragged_fixed_chunk_acquire)
            {
                signal_chunk_rows = std::max(
                    1,
                    get_int_from_env_local(
                        "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_CHUNK_ACQUIRE_ROWS",
                        2048));
            }
        }
        if (!ragged_ring_chunk && !ragged_fixed_chunk_acquire &&
            signal_chunk_rows > local_m)
        {
            signal_chunk_rows = local_m;
        }
        const bool native_kn_tile_ready = use_native_kn_chunk &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_TILE_READY", false);
        if (native_kn_tile_ready)
        {
            // The first protocol milestone uses one signal per owner rank.
            // This keeps the tile dependency exact for compact rank ranges
            // while avoiding a signal/DMA launch for every 256-row tile.
            signal_chunk_rows = local_m;
        }
        if (use_v36_rank_chunk_signal && !ragged_ring_chunk &&
            !ragged_fixed_chunk_acquire)
        {
            signal_chunk_rows = local_m;
        }
        const int chunk_split_rows =
            (ragged_ring_chunk || ragged_fixed_chunk_acquire)
            ? max_rank_m_rows
            : local_m;
        const int chunks_per_rank =
            (chunk_split_rows + signal_chunk_rows - 1) / signal_chunk_rows;
        const bool requested_chunk_signal =
            use_v11_chunk_signal || ragged_ring_chunk ||
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_CHUNK_SIGNAL", false);
        const bool use_chunk_signal =
            use_rank_signal &&
            (requested_chunk_signal || use_native_kn_chunk ||
             use_nn_direct_chunk_signal || use_v36_rank_chunk_signal) &&
            (use_source_dsread_n128 || use_v11_chunk_signal ||
             use_native_kn_chunk || use_nn_direct_chunk_signal ||
             use_v36_rank_chunk_signal || ragged_ring_chunk);
        if (requested_chunk_signal && !ragged_m_ && !use_native_kn_chunk)
        {
            FTRAIN_CHECK(use_rank_signal &&
                       (use_source_dsread_n128 || use_v11_chunk_signal ||
                        use_nn_direct_chunk_signal))
                << "AGGroupGemm chunk signal requires rank-tile BF16 "
                   "dsread256-buffer-LDS-n128 or V11 ASM route";
        }
        const int chunk_signal_slots =
            world_size_ * chunks_per_rank * 2;
        const int payload_signal_capacity = signal_capacity_ -
            (uses_ring_reuse_credit() ? 1 : 0);
        FTRAIN_CHECK(!use_chunk_signal ||
                   chunk_signal_slots <= payload_signal_capacity)
            << "AGGroupGemm chunk signal needs "
            << chunk_signal_slots
            << " signal slots, payload capacity=" << payload_signal_capacity;
        const bool use_problem_list =
            (use_chunk_signal || use_native_kn_chunk) &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_RANK_TILE_PROBLEM_LIST",
                              true);
        const bool signal_local_chunks_early =
            use_chunk_signal &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_LOCAL_CHUNK_READY",
                              use_v11_chunk_signal || use_native_kn_chunk ||
                                  use_nn_direct_chunk_signal ||
                                  use_v36_rank_chunk_signal);
        const bool v11_chunk_bootstrap =
            use_v11_chunk_signal && signal_local_chunks_early &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_V11_CHUNK_BOOTSTRAP", true);
        const bool strict_rank_tile =
            requested_strict_rank_tile ||
            (use_problem_list && !use_v11_chunk_signal && !compact_kn_route);
        const bool use_push_copy =
            !use_chunk_signal && (ragged_m_ || requested_push_copy);
        const bool v36_stage_ordered_requested =
            get_bool_from_env_with_legacy_local(
                "FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE",
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_STAGE_ORDERED",
                false);
            const int v36_n_tiles = (output_dim_ + 255) / 256;
        const bool v40_ring_env_requested = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_V40_TRUE_RING", false);
        const bool v40_true_ring_requested = v36_n_tiles > 0 &&
            v40_ring_env_requested;
        const bool v36_dynamic = v36_stage_ordered_requested &&
            (!v40_true_ring_requested ||
             (ragged_m_ && !ragged_v40_scheduler)) &&
            v36_n_tiles > 0 && v36_n_tiles <= 64 &&
            get_bool_from_env_with_legacy_local(
                "FTRAIN_AG_GROUP_GEMM_RANK_TILE_DYNAMIC",
                "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_DYNAMIC",
                true);
        const bool v36_stage_ordered =
            v36_stage_ordered_requested &&
            (v36_n_tiles == 8 || v36_dynamic ||
             v40_true_ring_requested);
        if (ragged_m_)
        {
            FTRAIN_CHECK(use_tiled && use_rank_signal && use_direct_mmac &&
                       use_mt256_native_contract && use_v36_topk &&
                       (v36_dynamic || ragged_v40_scheduler) &&
                       v36_stage_ordered &&
                       !get_bool_from_env_with_legacy_local(
                           "FTRAIN_AG_GROUP_GEMM_RANK_TILE_DIRECT_OUTPUT",
                           "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_DIRECT_OUTPUT",
                           false))
                << "ragged-M currently requires the BF16 V36 dynamic native "
                   "rank-signal WGM8 route without direct output";
        }
        const bool v36_v5_fallback =
            v36_stage_ordered_requested && !v36_stage_ordered;
        const bool v40_true_ring = v36_stage_ordered && !v36_dynamic &&
            v40_true_ring_requested;
        const bool v40_partial_ready = v40_true_ring && get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY", false);
        FTRAIN_CHECK(!use_v36_rank_chunk_signal ||
                   (v40_true_ring && v40_partial_ready) ||
                   ragged_ring_chunk)
            << "V36 rank-chunk readiness requires V40 partial-ready ring";
        const bool force_physical_signal_layout = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_FORCE_PHYSICAL_SIGNAL_LAYOUT", false);
        const bool use_stage_signal_layout =
            !force_physical_signal_layout && use_push_copy &&
            !v36_v5_fallback &&
            (v36_stage_ordered ||
             get_bool_from_env(
                 "FTRAIN_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT", false));
        const bool stage_signal_zigzag = use_stage_signal_layout &&
            !route_spec_.nk_new_a2a_ring &&
            !v40_true_ring && (v36_stage_ordered ||
             get_bool_from_env(
                 "FTRAIN_AG_GROUP_GEMM_PUSH_ZIGZAG_ORDER", true));
        const bool use_copy_nocu =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_COPY_NOCU", false);
        const int active_signal_chunk_rows =
            use_chunk_signal ? signal_chunk_rows : 0;
        const int active_chunks_per_rank =
            use_chunk_signal ? chunks_per_rank : 0;
        const bool profile_requested =
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_PROFILE", false);
        const bool profile_all_ranks = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_PROFILE_ALL_RANKS", false);
        const int requested_profile_skip =
            get_int_from_env_local("FTRAIN_AG_GROUP_GEMM_PROFILE_SKIP", 0);
        const int profile_skip =
            requested_profile_skip > 0 ? requested_profile_skip : 0;
        const int profile_limit =
            get_int_from_env_local("FTRAIN_AG_GROUP_GEMM_PROFILE_ITERS", 4);
        const bool profile_rank = rank_ == 0 || profile_all_ranks;
        int profile_call_index = -1;
        if (profile_requested && profile_rank)
        {
            profile_call_index = profile_forward_count_++;
        }
        const bool profile_this_forward =
            profile_requested && profile_rank &&
            profile_call_index >= profile_skip &&
            profile_call_index < profile_skip + profile_limit;
        hipEvent_t prof_start = nullptr;
        hipEvent_t prof_after_pack = nullptr;
        hipEvent_t prof_after_ag = nullptr;
        hipEvent_t prof_after_meta = nullptr;
        hipEvent_t prof_after_gemm = nullptr;
        hipEvent_t prof_remote_done = nullptr;
        hipEvent_t prof_end = nullptr;
        if (profile_this_forward)
        {
            CUDA_CHECK(hipEventCreate(&prof_start));
            CUDA_CHECK(hipEventCreate(&prof_after_pack));
            CUDA_CHECK(hipEventCreate(&prof_after_ag));
            CUDA_CHECK(hipEventCreate(&prof_after_meta));
            CUDA_CHECK(hipEventCreate(&prof_after_gemm));
            CUDA_CHECK(hipEventCreate(&prof_remote_done));
            CUDA_CHECK(hipEventCreate(&prof_end));
            CUDA_CHECK(hipEventRecord(prof_start, stream));
        }

        if (!logged_route_ &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_DEBUG_ROUTE", false) &&
            rank_ == 0)
        {
            FTRAIN_LOG_INFO("AGGroupGemm route M=%d N=%d K=%d E=%d topk=%d "
                          "world=%d hip_overlap=1 hip_mmac=%d rank_tile=%d "
                          "strict_rank_tile=%d lds_mmac=%d pack_weight=%d "
                          "push_copy=%d copy_nocu=%d shmem_b=%d "
                          "tile_ready=%d source_dsread_n128=%d "
                          "native_kn_tile_ready=%d "
                          "dsread_fragment_layout=%d "
                          "dsread_swizzle=%d "
                          "wg8_fragment=%d "
                          "wg8_m32=%d wg16_m32_n64=%d "
                          "wg16_pair_b=%d chunk_signal=%d "
                          "problem_list=%d signal_chunk_rows=%d "
                          "chunks_per_rank=%d "
                          "local_chunk_ready=%d chunk_bootstrap=%d "
                          "true_ring=%d partial_ready=%d "
                          "high_priority_comm=%d "
                          "v36_stage_ordered_requested=%d v36_dynamic=%d "
                          "v36_stage_ordered=%d v36_v5_fallback=%d "
                          "stage_signal_layout=%d stage_signal_zigzag=%d "
                          "source_impl=1",
                          full_m_,
                          output_dim_,
                          reduction_dim_,
                          num_experts_,
                          topk_,
                          world_size_,
                          use_tiled ? 1 : 0,
                          use_rank_signal ? 1 : 0,
                          strict_rank_tile ? 1 : 0,
                          use_lds_mmac ? 1 : 0,
                          use_packed_weight ? 1 : 0,
                          use_push_copy ? 1 : 0,
                          use_copy_nocu ? 1 : 0,
                          use_mmac_shmem_b ? 1 : 0,
                          use_tile_ready ? 1 : 0,
                          use_source_dsread_n128 ? 1 : 0,
                          native_kn_tile_ready ? 1 : 0,
                          use_dsread_fragment_layout ? 1 : 0,
                          use_dsread_swizzle ? 1 : 0,
                          use_wg8_fragment ? 1 : 0,
                          use_wg8_m32 ? 1 : 0,
                          use_wg16_m32_n64 ? 1 : 0,
                          use_wg16_pair_b ? 1 : 0,
                          use_chunk_signal ? 1 : 0,
                          use_problem_list ? 1 : 0,
                          active_signal_chunk_rows,
                          active_chunks_per_rank,
                          signal_local_chunks_early ? 1 : 0,
                          v11_chunk_bootstrap ? 1 : 0,
                          v40_true_ring ? 1 : 0,
                          v40_partial_ready ? 1 : 0,
                          comm_stream_high_priority_ ? 1 : 0,
                          v36_stage_ordered_requested ? 1 : 0,
                          v36_dynamic ? 1 : 0,
                          v36_stage_ordered ? 1 : 0,
                          v36_v5_fallback ? 1 : 0,
                          use_stage_signal_layout ? 1 : 0,
                          stage_signal_zigzag ? 1 : 0);
            logged_route_ = true;
        }

        const void *gemm_weight = weight.data_ptr();
        bool weight_nk = input_weight_nk;
        bool weight_repacked = false;
        bool weight_pack_in_flight = false;
        if (use_packed_weight)
        {
            // Megatron's distributed optimizer updates model buffers through
            // Tensor::data, which does not reliably bump the Parameter version.
            // Repack trainable weights unless the caller explicitly guarantees
            // that their contents remain immutable.
            const bool force_training_repack = weight.requires_grad() &&
                !get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_ASSUME_TRAINING_WEIGHT_IMMUTABLE",
                    false);
            const int64_t weight_version =
                weight.version();
            if (!packed_weight_.defined())
            {
                packed_weight_ = empty(
                    {num_experts_, output_dim_, reduction_dim_},
                                              weight.options());
                packed_weight_src_ = nullptr;
                packed_weight_version_ = -1;
            }
            if (packed_weight_src_ != weight.data_ptr() ||
                packed_weight_version_ != weight_version ||
                force_training_repack)
            {
                CUDA_CHECK(hipEventRecord(weight_pack_start_event_, stream));
                CUDA_CHECK(hipStreamWaitEvent(weight_pack_stream_.get(),
                                              weight_pack_start_event_,
                                              0));
                ag_group_gemm_pack_weight_impl(weight.data_ptr(),
                                               packed_weight_.data_ptr(),
                                               num_experts_,
                                               reduction_dim_,
                                               output_dim_,
                                               input_dtype_,
                                               weight_pack_stream_.get());
                CUDA_CHECK(hipEventRecord(weight_pack_done_event_,
                                          weight_pack_stream_.get()));
                weight_repacked = true;
                weight_pack_in_flight = true;
                packed_weight_src_ = weight.data_ptr();
                packed_weight_version_ = weight_version;
            }
            gemm_weight = packed_weight_.data_ptr();
            weight_nk = true;
        }
        if (profile_this_forward)
        {
            CUDA_CHECK(hipEventRecord(
                prof_after_pack,
                weight_pack_in_flight ? weight_pack_stream_.get() : stream));
        }

        const int signal_value = launch_rank_signal_allgather(
            input,
            stream,
            use_push_copy,
            use_copy_nocu,
            use_chunk_signal,
            active_signal_chunk_rows,
            signal_local_chunks_early,
            v11_chunk_bootstrap,
            use_native_kn_weight,
            use_native_kn_chunk,
            use_v36_rank_chunk_signal);
        int v40_bootstrap_stages = v40_true_ring
            ? get_int_from_env_local(
                  "FTRAIN_AG_GROUP_GEMM_V40_BOOTSTRAP_STAGES", 0)
            : 0;
        const char *prefetch_stages_env =
            std::getenv("FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_PREFETCH_STAGES");
        const bool prefetch_stages_explicit =
            prefetch_stages_env != nullptr && prefetch_stages_env[0] != '\0';
        int v40_partial_ready_prefetch_stages = v40_partial_ready
            ? get_int_from_env_local(
                  "FTRAIN_AG_GROUP_GEMM_V40_PARTIAL_READY_PREFETCH_STAGES", 0)
            : 0;
        const bool v40_gate_gemm = v40_true_ring && !v40_partial_ready &&
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_V40_GATE_GEMM", false);
        // Ragged native-KN ring currently uses whole-owner completion.  Gate
        // the GEMM until the ring has published every owner segment; this is
        // the correctness-safe contract.  Fixed-M V40 keeps its own policy.
        // The first ragged-ring implementation intentionally gates the GEMM on
        // remote_ready.  Variable-length whole-owner overlap produced stale
        // reads on gfx936; a chunk-ready body/protocol is required before
        // enabling safe overlap.
        // With chunk-ready publication the body can start before remote_ready;
        // preserve the safe whole-owner contract unless overlap was explicitly
        // requested through the established WAIT_REMOTE_READY=0 knob.
        const bool ragged_fixed_owner_signals = ragged_m_ &&
            use_native_kn_weight && v40_true_ring && get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_RING_OWNER_SIGNALS", false);
        if (ragged_fixed_owner_signals && !prefetch_stages_explicit)
        {
            // Choose the ragged launch head-start from transport depth.  ETP8
            // benefits from waiting for the first acquired remote owner; ETP4
            // has only three remote stages and is faster with no launch gate,
            // letting the dynamic body begin on local/early owners while the
            // remaining ring stages continue.
            // ETP2 has one remote owner, so stage1 would gate on the complete
            // transport; measured runs also prefer no launch gate.
            if (world_size_ >= 8)
            {
                v40_partial_ready_prefetch_stages = 1;
            }
            else if (world_size_ == 4 || world_size_ == 2)
            {
                v40_partial_ready_prefetch_stages = 0;
            }
            else
            {
                v40_partial_ready_prefetch_stages = 0;
            }
        }
        const bool ragged_ring_gate =
            ragged_m_ && use_native_kn_weight && v40_ring_env_requested &&
            get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_RAGGED_RING_WAIT_REMOTE_READY", true);
        const bool push_ring_gate = get_bool_from_env(
            "FTRAIN_AG_GROUP_GEMM_PUSH_RING_GATE", false) || ragged_ring_gate;
        v40_bootstrap_stages = std::max(
            0, std::min(v40_bootstrap_stages, world_size_ - 1));
        v40_partial_ready_prefetch_stages = std::max(
            0,
            std::min(v40_partial_ready_prefetch_stages, world_size_ - 1));
        // Startup gate for the body launch.  The whole-segment ring waits for
        // its first prefetch stage slots.  The chunk-ready transport instead
        // waits for this rank's own chunks (slot = rank * chunks + chunk), so
        // the body starts with the local owner segment resident instead of
        // launching straight into a full spin.  Measured ~0.28..0.64 ms faster
        // than launching immediately; CHUNK_READY_CHUNKS picks how many of the
        // local chunks to require (0 = all).
        int v40_initial_wait_slot = -1;
        if (v40_partial_ready && !use_chunk_signal)
        {
            if (ragged_fixed_owner_signals)
            {
                // Ragged fixed-ring uses disjoint raw/internal/body signal
                // namespaces.  If a head-start is requested, wait for the
                // acquired body slot of that ring stage, not the private
                // internal forwarding slot.
                v40_initial_wait_slot =
                    v40_partial_ready_prefetch_stages > 0
                        ? get_int_from_env_local(
                              "FTRAIN_AG_GROUP_GEMM_RAGGED_FIXED_BODY_SLOT_BASE",
                              24) +
                              (rank_ + v40_partial_ready_prefetch_stages) %
                                  world_size_
                        : -1;
            }
            else
            {
                v40_initial_wait_slot =
                    v40_partial_ready_prefetch_stages > 0
                        ? v40_partial_ready_prefetch_stages
                        : -1;
            }
        }
        else if (use_chunk_signal)
        {
            if (ragged_fixed_chunk_acquire)
            {
                v40_initial_wait_slot = -1;
            }
            if (!ragged_fixed_chunk_acquire &&
                active_chunks_per_rank > 0 &&
                active_signal_chunk_rows > 0 &&
                !rank_m_sizes_.empty())
            {
                int local_chunks = static_cast<int>(
                    (static_cast<int64_t>(rank_m_sizes_[rank_]) +
                     active_signal_chunk_rows - 1) /
                    active_signal_chunk_rows);
                local_chunks = std::max(
                    1, std::min(local_chunks, active_chunks_per_rank));
                int gate_chunks = get_int_from_env_local(
                    "FTRAIN_AG_GROUP_GEMM_CHUNK_READY_CHUNKS", 0);
                if (gate_chunks <= 0 || gate_chunks > local_chunks)
                {
                    gate_chunks = local_chunks;
                }
                v40_initial_wait_slot = ragged_fixed_chunk_acquire
                    ? get_int_from_env_local(
                          "FTRAIN_AG_GROUP_GEMM_RAGGED_CHUNK_SLOT_BASE", 32) +
                          gate_chunks - 1
                    : rank_ * active_chunks_per_rank + gate_chunks - 1;
            }
        }
        else
        {
            v40_initial_wait_slot = v40_bootstrap_stages > 0
                ? v40_bootstrap_stages
                : -1;
        }
        if (route_spec_.nk_new_a2a_ring)
        {
            const int prefetch_stages = get_int_from_env_local(
                "FTRAIN_AG_GROUP_GEMM_NK_NEW_RING_PREFETCH_STAGES", 0);
            FTRAIN_CHECK(prefetch_stages >= 0 && prefetch_stages < world_size_);
            if (prefetch_stages > 0)
                v40_initial_wait_slot = world_size_ + prefetch_stages;
        }
        if (enk_ring_chunk_ready_ || route_spec_.kn_ring_chunk_ready)
            v40_initial_wait_slot = -1;
        if (v40_gate_gemm || push_ring_gate)
        {
            CUDA_CHECK(hipStreamWaitEvent(stream, remote_ready_event_, 0));
        }
        else if (signal_value > 0 && v40_initial_wait_slot > 0)
        {
            CUDA_CHECK(hipStreamWaitValue32(
                stream,
                static_cast<int *>(barrier_buffer_.data_ptr()) +
                    v40_initial_wait_slot,
                static_cast<uint32_t>(signal_value),
                hipStreamWaitValueGte));
        }
        if (profile_this_forward)
        {
            CUDA_CHECK(hipEventRecord(prof_after_ag, stream));
            CUDA_CHECK(hipEventRecord(prof_remote_done, comm_stream_));
        }
        const int *rank_signals =
            static_cast<const int *>(barrier_buffer_.data_ptr());
        bool profile_metadata_cache_hit = false;

        if (use_tiled)
        {
            const bool use_metadata_cache = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_CK_ENABLE_METADATA_CACHE", true);
            const bool assume_metadata_immutable = get_bool_from_env(
                "FTRAIN_AG_GROUP_GEMM_CK_METADATA_CACHE_ASSUME_IMMUTABLE", false);
            const void *topk_ptr = full_topk_ids.data_ptr();
            const int64_t topk_version =
                full_topk_ids.version();
            const bool metadata_cache_hit =
                use_metadata_cache && metadata_cache_valid_ &&
                metadata_topk_tensor_.is_same(full_topk_ids) &&
                metadata_topk_src_ == topk_ptr &&
                metadata_topk_dtype_ == full_topk_ids.scalar_type() &&
                metadata_strict_rank_tile_ == strict_rank_tile &&
                metadata_split_local_rank_ == requested_split_local_rank &&
                metadata_isolate_local_owner_ == isolate_local_owner &&
                metadata_build_problem_info_ == use_problem_list &&
                metadata_stage_signal_layout_ == use_stage_signal_layout &&
                metadata_stage_signal_zigzag_ == stage_signal_zigzag &&
                metadata_signal_chunk_rows_ == active_signal_chunk_rows &&
                metadata_chunks_per_rank_ == active_chunks_per_rank &&
                (assume_metadata_immutable ||
                 metadata_topk_version_ == topk_version);
            profile_metadata_cache_hit = metadata_cache_hit;
            if (!metadata_cache_hit)
            {
                ag_group_gemm_prepare_metadata_impl(
                    full_topk_ids.data_ptr(),
                    sorted_token_ids_.data_ptr<int>(),
                    sorted_gather_index_.data_ptr<int>(),
                    expert_rank_counts_.data_ptr<int>(),
                    expert_rank_counters_.data_ptr<int>(),
                    expert_rank_offsets_.data_ptr<int>(),
                    global_expert_offsets_.data_ptr<int>(),
                    rank_cursors_.data_ptr<int>(),
                    rank_tile_info_.data_ptr<int>(),
                    tile_experts_.data_ptr<int>(),
                    tile_starts_.data_ptr<int>(),
                    tile_rank_starts_.data_ptr<int>(),
                    tile_rank_ends_.data_ptr<int>(),
                    use_problem_list ? rank_tile_problem_info_.data_ptr<int>()
                                     : nullptr,
                    ntiles_.data_ptr<int>(),
                    ragged_m_ ? d_rank_m_offsets_ : nullptr,
                    full_m_,
                    topk_,
                    num_experts_,
                    world_size_,
                    rank_,
                    sorted_capacity_,
                    max_tiles_,
                    active_signal_chunk_rows,
                    active_chunks_per_rank,
                    strict_rank_tile,
                    requested_split_local_rank,
                    use_problem_list,
                    use_stage_signal_layout,
                    stage_signal_zigzag,
                    full_topk_ids.scalar_type(),
                    stream);
                metadata_topk_tensor_ = full_topk_ids;
                metadata_cache_valid_ = true;
                metadata_topk_src_ = topk_ptr;
                metadata_topk_version_ = topk_version;
                metadata_topk_dtype_ = full_topk_ids.scalar_type();
                metadata_strict_rank_tile_ = strict_rank_tile;
                metadata_split_local_rank_ = requested_split_local_rank;
                metadata_isolate_local_owner_ = isolate_local_owner;
                metadata_build_problem_info_ = use_problem_list;
                metadata_stage_signal_layout_ = use_stage_signal_layout;
                metadata_stage_signal_zigzag_ = stage_signal_zigzag;
                metadata_signal_chunk_rows_ = active_signal_chunk_rows;
                metadata_chunks_per_rank_ = active_chunks_per_rank;
            }
            if (get_bool_from_env("FTRAIN_AG_GROUP_GEMM_DEBUG_ROUTED_BASE", false) &&
                rank_ == 0)
            {
                int host_ntiles = 0;
                std::vector<int> host_info(static_cast<size_t>(max_tiles_) * 5,
                                           0);
                // Keep enough of the row map to validate the first strict
                // expert/rank tiles.  The previous 64-entry snapshot printed
                // -1 for every later tile and could not distinguish a real
                // discontinuity from a debug-buffer truncation.
                std::vector<int> host_gather(sorted_capacity_, -1);
                CUDA_CHECK(hipMemcpyAsync(
                    &host_ntiles,
                    ntiles_.data_ptr(),
                    sizeof(int),
                    hipMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(hipMemcpyAsync(
                    host_info.data(),
                    rank_tile_info_.data_ptr(),
                    host_info.size() * sizeof(int),
                    hipMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(hipMemcpyAsync(
                    host_gather.data(),
                    sorted_gather_index_.data_ptr(),
                    host_gather.size() * sizeof(int),
                    hipMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(hipStreamSynchronize(stream));
                printf("[moe] routed debug ntiles=%d gather0=%d\\n",
                       host_ntiles,
                       host_gather.empty() ? -1 : host_gather[0]);
                for (int tile = 0; tile < std::min(host_ntiles, 8); ++tile)
                {
                    const int *info = host_info.data() + tile * 5;
                    const int start = info[1];
                    const int first =
                        start >= 0 && start < static_cast<int>(host_gather.size())
                            ? host_gather[start]
                            : -1;
                    int discontinuities = 0;
                    if (start >= 0 && info[2] > 1 &&
                        start + info[2] <=
                            static_cast<int>(host_gather.size()))
                    {
                        for (int row = 1; row < info[2]; ++row)
                        {
                            discontinuities +=
                                host_gather[start + row] !=
                                host_gather[start] + row;
                        }
                    }
                    printf("[moe] routed tile=%d expert=%d start=%d rows=%d "
                           "owner=%d..%d first=%d discontinuities=%d\\n",
                           tile,
                           info[0],
                           start,
                           info[2],
                           info[3],
                           info[4],
                           first,
                           discontinuities);
                }
                int total_discontinuities = 0;
                int first_bad_tile = -1;
                int first_bad_row = -1;
                for (int tile = 0; tile < host_ntiles; ++tile)
                {
                    const int *info = host_info.data() + tile * 5;
                    const int start = info[1];
                    const int rows = info[2];
                    if (start < 0 || rows <= 1 ||
                        start + rows > static_cast<int>(host_gather.size()))
                    {
                        continue;
                    }
                    for (int row = 1; row < rows; ++row)
                    {
                        if (host_gather[start + row] !=
                            host_gather[start] + row)
                        {
                            ++total_discontinuities;
                            if (first_bad_tile < 0)
                            {
                                first_bad_tile = tile;
                                first_bad_row = row;
                            }
                        }
                    }
                }
                printf("[moe] routed debug total_discontinuities=%d "
                       "first_bad=(%d,%d)\n",
                       total_discontinuities,
                       first_bad_tile,
                       first_bad_row);
            }
            if (get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_DEBUG_OWNER_SPANS", false) &&
                (rank_ == 0 || get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_DEBUG_OWNER_SPANS_ALL_RANKS", false)))
            {
                int host_ntiles = 0;
                std::vector<int> host_info(
                    static_cast<size_t>(max_tiles_) * 5, 0);
                CUDA_CHECK(hipMemcpyAsync(
                    &host_ntiles,
                    ntiles_.data_ptr(),
                    sizeof(int),
                    hipMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(hipMemcpyAsync(
                    host_info.data(),
                    rank_tile_info_.data_ptr(),
                    host_info.size() * sizeof(int),
                    hipMemcpyDeviceToHost,
                    stream));
                CUDA_CHECK(hipStreamSynchronize(stream));
                std::vector<int> span_hist(static_cast<size_t>(world_size_) + 1,
                                           0);
                std::vector<int> owner_hist(static_cast<size_t>(world_size_), 0);
                for (int tile = 0; tile < host_ntiles; ++tile)
                {
                    const int *info = host_info.data() + tile * 5;
                    const int begin = std::max(0, info[3]);
                    const int end = std::min(world_size_ - 1, info[4]);
                    const int span = end >= begin ? end - begin + 1 : 0;
                    if (span >= 0 && span <= world_size_)
                    {
                        ++span_hist[static_cast<size_t>(span)];
                    }
                    if (begin == end && begin >= 0 && begin < world_size_)
                    {
                        ++owner_hist[static_cast<size_t>(begin)];
                    }
                }
                printf("[moe] owner spans ntiles=%d hist:", host_ntiles);
                for (int span = 0; span <= world_size_; ++span)
                {
                    if (span_hist[static_cast<size_t>(span)] != 0)
                    {
                        printf(" %d:%d", span,
                               span_hist[static_cast<size_t>(span)]);
                    }
                }
                printf(" owners:");
                for (int owner = 0; owner < world_size_; ++owner)
                {
                    printf(" %d:%d", owner,
                           owner_hist[static_cast<size_t>(owner)]);
                }
                printf(" rank=%d\n", rank_);
            }
            if (profile_this_forward)
            {
                CUDA_CHECK(hipEventRecord(prof_after_meta, stream));
            }

            last_signal_value_ = signal_value;
            last_forward_used_rank_signal_ = use_rank_signal;
            CUDA_CHECK(hipEventRecord(wgrad_metadata_ready_event_, stream));
            last_forward_valid_ = true;

            if (!use_rank_signal)
            {
                if (use_push_copy)
                {
                    ag_group_gemm_wait_rank_signals_impl(
                        rank_signals, world_size_, signal_value, stream);
                }
                else
                {
                    CUDA_CHECK(
                        hipStreamWaitEvent(stream, remote_ready_event_, 0));
                }
            }

            const uint64_t v36_direct_output_bytes =
                static_cast<uint64_t>(full_m_) *
                static_cast<uint64_t>(topk_) *
                static_cast<uint64_t>(output_dim_) * sizeof(uint16_t);
            const bool use_v36_native_direct_n_chunk_done =
                use_tiled && use_direct_mmac && use_mt256_native_contract &&
                input_dtype_ == ScalarType::BFloat16 && v36_dynamic &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false) &&
                get_bool_from_env(
                    "FTRAIN_AG_GROUP_GEMM_ASM_MT256_WGM8", false) &&
                get_bool_from_env_with_legacy_local(
                    "FTRAIN_AG_GROUP_GEMM_RANK_TILE_DIRECT_OUTPUT",
                    "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_DIRECT_OUTPUT",
                    false) &&
                get_bool_from_env_with_legacy_local(
                    "FTRAIN_AG_GROUP_GEMM_RANK_TILE_DIRECT_NCHUNK_SIGNAL",
                    "FTRAIN_AG_GROUP_GEMM_ASM_MT256_V36_DIRECT_NCHUNK_SIGNAL",
                    true) &&
                v36_direct_output_bytes <= 0xfffffffeULL;
            const bool use_n_chunk_done =
                (use_tiled && use_direct_mmac && use_lds_mmac &&
                 input_dtype_ == ScalarType::BFloat16) ||
                use_v36_native_direct_n_chunk_done;
            if (use_rank_signal && signal_value == 1)
            {
                CUDA_CHECK(hipMemsetAsync(rank_tile_ready_flags_.data_ptr(),
                                          0,
                                          sizeof(int) * max_tiles_,
                                          stream));
            }
            if (use_n_chunk_done)
            {
                CUDA_CHECK(hipMemsetAsync(n_chunk_done_counts_.data_ptr(),
                                          0,
                                          sizeof(int) * n_chunk_done_groups_,
                                          stream));
                CUDA_CHECK(hipMemsetAsync(n_chunk_done_flags_.data_ptr(),
                                          0,
                                          sizeof(int) * n_chunk_done_groups_,
                                          stream));
            }
            const int launch_max_tiles =
                strict_rank_tile ? max_tiles_ : aggregate_max_tiles_;
            if (weight_pack_in_flight)
            {
                CUDA_CHECK(
                    hipStreamWaitEvent(stream, weight_pack_done_event_, 0));
            }
            // Native-KN consumes the Megatron-expanded expert-major buffer
            // directly.  Its strict rank-tile row map assumes that every
            // rank's copied segment is globally visible before the first
            // tile starts.  The per-rank signal is sufficient for the
            // historical v36 scheduler, but on gfx936 a signal observed by
            // the hand-written KN body can race the peer-copy visibility.
            // Fence the KN launch against the aggregate copy event; this
            // preserves overlap with metadata construction and only removes
            // the unsafe tail of copy/GEMM overlap.
            if (use_native_kn_weight && use_push_copy &&
                (get_bool_from_env(
                     "FTRAIN_AG_GROUP_GEMM_NATIVE_KN_WAIT_REMOTE_READY", false) ||
                 compact_kn_route))
            {
                CUDA_CHECK(hipStreamWaitEvent(stream, remote_ready_event_, 0));
            }
            if (use_native_kn_weight)
            {
                ag_group_gemm_native_kn_tiled_forward_impl(
                    input_buffer_.data_ptr(),
                    weight.data_ptr(),
                    sorted_token_ids_.data_ptr<int>(),
                    sorted_gather_index_.data_ptr<int>(),
                    rank_tile_info_.data_ptr<int>(),
                    tile_experts_.data_ptr<int>(),
                    tile_starts_.data_ptr<int>(),
                    tile_rank_starts_.data_ptr<int>(),
                    tile_rank_ends_.data_ptr<int>(),
                    (use_native_kn_chunk || use_nn_direct_chunk_signal ||
                     use_v36_rank_chunk_signal)
                        ? rank_tile_problem_info_.data_ptr<int>()
                        : nullptr,
                    ntiles_.data_ptr<int>(),
                    use_rank_signal ? rank_signals : nullptr,
                    use_rank_signal
                        ? rank_tile_ready_flags_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_counts_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_flags_.data_ptr<int>()
                        : nullptr,
                    out.data_ptr(),
                    full_m_,
                    output_dim_,
                    reduction_dim_,
                    topk_,
                    world_size_,
                    rank_,
                    launch_max_tiles,
                    signal_value,
                    active_signal_chunk_rows,
                    active_chunks_per_rank,
                    ragged_m_,
                    input_dtype_,
                    stream,
                    route_spec_.kn_ring_chunk_ready ? d_rank_m_offsets_ : nullptr,
                    active_signal_chunk_rows);
            }
            else if ((enk_ring_chunk_ready_ || supports_ragged_chunk_a2a()) && ragged_m_)
            {
                ag_group_gemm_enk_ring_ragged_forward_impl(
                    input_buffer_.data_ptr(),
                    gemm_weight,
                    sorted_token_ids_.data_ptr<int>(),
                    sorted_gather_index_.data_ptr<int>(),
                    rank_tile_info_.data_ptr<int>(),
                    tile_experts_.data_ptr<int>(),
                    tile_starts_.data_ptr<int>(),
                    tile_rank_starts_.data_ptr<int>(),
                    tile_rank_ends_.data_ptr<int>(),
                    ntiles_.data_ptr<int>(),
                    use_rank_signal ? rank_signals : nullptr,
                    use_rank_signal
                        ? rank_tile_ready_flags_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_counts_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_flags_.data_ptr<int>()
                        : nullptr,
                    out.data_ptr(),
                    full_m_,
                    output_dim_,
                    reduction_dim_,
                    topk_,
                    world_size_,
                    rank_,
                    launch_max_tiles,
                    signal_value,
                    active_signal_chunk_rows,
                    active_chunks_per_rank,
                    true,
                    input_dtype_,
                    stream,
                    d_rank_m_offsets_,
                    supports_ragged_chunk_a2a() ? 0 :
                        (*std::max_element(rank_m_sizes_.begin(), rank_m_sizes_.end()) + 7) / 8);
            }
            else if (use_mt256_native_contract)
            {
                ag_group_gemm_legacy_tn_tiled_forward_impl(
                    input_buffer_.data_ptr(),
                    gemm_weight,
                    sorted_token_ids_.data_ptr<int>(),
                    sorted_gather_index_.data_ptr<int>(),
                    rank_tile_info_.data_ptr<int>(),
                    tile_experts_.data_ptr<int>(),
                    tile_starts_.data_ptr<int>(),
                    tile_rank_starts_.data_ptr<int>(),
                    tile_rank_ends_.data_ptr<int>(),
                    ntiles_.data_ptr<int>(),
                    use_rank_signal ? rank_signals : nullptr,
                    use_rank_signal
                        ? rank_tile_ready_flags_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_counts_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_flags_.data_ptr<int>()
                        : nullptr,
                    out.data_ptr(),
                    full_m_,
                    output_dim_,
                    reduction_dim_,
                    topk_,
                    world_size_,
                    rank_,
                    launch_max_tiles,
                    signal_value,
                    active_signal_chunk_rows,
                    active_chunks_per_rank,
                    false,
                    input_dtype_,
                    stream);
            }
            else
            {
                ag_group_gemm_tiled_forward_impl(
                    input_buffer_.data_ptr(),
                    gemm_weight,
                    sorted_token_ids_.data_ptr<int>(),
                    sorted_gather_index_.data_ptr<int>(),
                    rank_tile_info_.data_ptr<int>(),
                    tile_experts_.data_ptr<int>(),
                    tile_starts_.data_ptr<int>(),
                    tile_rank_starts_.data_ptr<int>(),
                    tile_rank_ends_.data_ptr<int>(),
                    use_problem_list
                        ? rank_tile_problem_info_.data_ptr<int>()
                        : nullptr,
                    ntiles_.data_ptr<int>(),
                    use_rank_signal ? rank_signals : nullptr,
                    use_rank_signal
                        ? rank_tile_ready_flags_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_counts_.data_ptr<int>()
                        : nullptr,
                    use_n_chunk_done
                        ? n_chunk_done_flags_.data_ptr<int>()
                        : nullptr,
                    out.data_ptr(),
                    full_m_,
                    output_dim_,
                    reduction_dim_,
                    topk_,
                    num_experts_,
                    world_size_,
                    rank_,
                    launch_max_tiles,
                    weight_nk,
                    profile_metadata_cache_hit,
                    ragged_m_,
                    signal_value,
                    active_signal_chunk_rows,
                    active_chunks_per_rank,
                    input_dtype_,
                    stream);
            }
            if (a2a_deferred_submit_)
            {
                auto submit = std::move(a2a_deferred_submit_);
                a2a_deferred_submit_ = nullptr;
                submit();
                if (profile_this_forward)
                    CUDA_CHECK(hipEventRecord(prof_remote_done, comm_stream_));
            }
            if (use_push_copy && use_rank_signal)
            {
                ag_group_gemm_wait_rank_signals_impl(
                    rank_signals, world_size_, signal_value, stream);
            }
        }
        else
        {
            if (use_push_copy)
            {
                ag_group_gemm_wait_rank_signals_impl(
                    rank_signals, world_size_, signal_value, stream);
            }
            else
            {
                CUDA_CHECK(hipStreamWaitEvent(stream, remote_ready_event_, 0));
            }
            ag_group_gemm_reference_forward_impl(input_buffer_.data_ptr(),
                                                 weight.data_ptr(),
                                                 full_topk_ids.data_ptr(),
                                                 out.data_ptr(),
                                                 full_m_,
                                                 output_dim_,
                                                 reduction_dim_,
                                                 num_experts_,
                                                 topk_,
                                                 input_dtype_,
                                                 full_topk_ids.scalar_type(),
                                                 stream);
            if (profile_this_forward)
            {
                CUDA_CHECK(hipEventRecord(prof_after_meta, stream));
            }
        }

        if (profile_this_forward)
        {
            CUDA_CHECK(hipEventRecord(prof_after_gemm, stream));
        }
        if (kn_chunk_transport_)
        {
            auto submit = std::move(kn_chunk_transport_);
            kn_chunk_transport_ = nullptr;
            submit();
            if (profile_this_forward)
                CUDA_CHECK(hipEventRecord(prof_remote_done, comm_stream_));
        }
        if (enk_chunk_transport_)
        {
            auto submit = std::move(enk_chunk_transport_);
            enk_chunk_transport_ = nullptr;
            submit();
            if (profile_this_forward)
                CUDA_CHECK(hipEventRecord(prof_remote_done, comm_stream_));
        }
        CUDA_CHECK(hipStreamWaitEvent(stream, remote_ready_event_, 0));
        if (profile_this_forward)
        {
            CUDA_CHECK(hipEventRecord(prof_end, stream));
            CUDA_CHECK(hipEventSynchronize(prof_end));
            CUDA_CHECK(hipEventSynchronize(prof_remote_done));
            float local_ms = 0.0f;
            float weight_pack_ms = 0.0f;
            float metadata_ms = 0.0f;
            float gemm_stream_ms = 0.0f;
            float remote_ms = 0.0f;
            float total_ms = 0.0f;
            CUDA_CHECK(hipEventElapsedTime(&local_ms,
                                           prof_start,
                                           prof_after_ag));
            CUDA_CHECK(hipEventElapsedTime(&weight_pack_ms,
                                           prof_start,
                                           prof_after_pack));
            CUDA_CHECK(hipEventElapsedTime(&metadata_ms,
                                           prof_after_ag,
                                           prof_after_meta));
            CUDA_CHECK(hipEventElapsedTime(&gemm_stream_ms,
                                           prof_after_meta,
                                           prof_after_gemm));
            CUDA_CHECK(hipEventElapsedTime(&remote_ms,
                                           prof_start,
                                           prof_remote_done));
            CUDA_CHECK(hipEventElapsedTime(&total_ms, prof_start, prof_end));
            FTRAIN_LOG_INFO("AGGroupGemm profile rank=%d iter=%d cache_hit=%d "
                          "weight_repacked=%d weight_pack_ms=%.3f "
                          "local_enqueue_ms=%.3f metadata_ms=%.3f "
                          "gemm_stream_ms=%.3f remote_ms=%.3f total_ms=%.3f",
                          rank_,
                          profile_call_index,
                          profile_metadata_cache_hit ? 1 : 0,
                          weight_repacked ? 1 : 0,
                          weight_pack_ms,
                          local_ms,
                          metadata_ms,
                          gemm_stream_ms,
                          remote_ms,
                          total_ms);
            hipEventDestroy(prof_start);
            hipEventDestroy(prof_after_pack);
            hipEventDestroy(prof_after_ag);
            hipEventDestroy(prof_after_meta);
            hipEventDestroy(prof_after_gemm);
            hipEventDestroy(prof_remote_done);
            hipEventDestroy(prof_end);
        }
        if (route_spec_.nk_new_a2a_chunk || enk_ring_chunk_ready_)
            CUDA_CHECK(hipEventRecord(a2a_consumer_done_event_, stream));
        return out.slice(0, 0, full_m_ * topk_);
    }

    Tensor backward_weight(Tensor full_routed_input)
    {
        auto grad_weight = backward_weight_async(std::move(full_routed_input));
        wait_wgrad();
        return grad_weight;
    }

    Tensor backward_weight_async(Tensor full_routed_input)
    {
        CHECK_INPUT(full_routed_input, input_dtype_);
        CHECK_2D(full_routed_input, full_m_ * topk_, output_dim_);
        FTRAIN_CHECK(input_dtype_ == ScalarType::BFloat16)
            << "AGGroupGemm backward_weight currently supports BF16";
        FTRAIN_CHECK(last_forward_valid_)
            << "backward_weight requires a completed forward launch on the "
               "same AGGroupGemm instance";
        FTRAIN_CHECK(!wgrad_pending_)
            << "backward_weight_async called while a previous wgrad is still "
               "pending; call wait_wgrad() first";

        auto grad_weight = empty(
            {num_experts_, output_dim_, reduction_dim_},
            full_routed_input.options());
        if (!wgrad_compact_routed_input_.defined())
        {
            wgrad_compact_routed_input_ = empty(
                {sorted_capacity_, output_dim_},
                full_routed_input.options());
            wgrad_compact_grad_output_ = empty(
                {sorted_capacity_, reduction_dim_},
                input_buffer_.options());
            wgrad_grouped_asm_args_ = empty(
                {num_experts_ * 196},
                full_routed_input.options().dtype(ScalarType::Byte));
        }
        CUDA_CHECK(hipStreamWaitEvent(
            wgrad_stream_, wgrad_metadata_ready_event_, 0));
        if (!last_forward_used_rank_signal_)
        {
            CUDA_CHECK(
                hipStreamWaitEvent(wgrad_stream_, remote_ready_event_, 0));
        }
        ag_group_gemm_backward_weight_impl(
            full_routed_input.data_ptr(),
            input_buffer_.data_ptr(),
            sorted_token_ids_.data_ptr<int>(),
            sorted_gather_index_.data_ptr<int>(),
            global_expert_offsets_.data_ptr<int>(),
            rank_cursors_.data_ptr<int>(),
            wgrad_compact_routed_input_.data_ptr(),
            wgrad_compact_grad_output_.data_ptr(),
            wgrad_grouped_asm_args_.data_ptr(),
            grad_weight.data_ptr(),
            sorted_capacity_,
            full_m_,
            output_dim_,
            reduction_dim_,
            num_experts_,
            topk_,
            world_size_,
            last_signal_value_,
            nullptr,
            0,
            input_dtype_,
            wgrad_stream_);
        CUDA_CHECK(hipEventRecord(wgrad_done_event_, wgrad_stream_));
        pending_wgrad_routed_input_ = std::move(full_routed_input);
        pending_wgrad_output_ = grad_weight;
        wgrad_pending_ = true;
        return grad_weight;
    }

    Tensor backward_weight_direct(Tensor routed_input,
                                         Tensor grad_output,
                                         Tensor full_topk_ids)
    {
        auto grad_weight = backward_weight_direct_async(
            std::move(routed_input),
            std::move(grad_output),
            std::move(full_topk_ids));
        wait_wgrad();
        return grad_weight;
    }

    Tensor backward_weight_direct_async(
        Tensor routed_input,
        Tensor grad_output,
        Tensor full_topk_ids)
    {
        return backward_weight_direct_async_impl(std::move(routed_input),
                                                 std::move(grad_output),
                                                 std::move(full_topk_ids),
                                                 std::nullopt,
                                                 0);
    }

    Tensor backward_weight_direct_chunked_async(
        Tensor routed_input,
        Tensor grad_output,
        Tensor full_topk_ids,
        Tensor output_chunk_ready_flags,
        int32_t output_chunk_size)
    {
        return backward_weight_direct_async_impl(
            std::move(routed_input),
            std::move(grad_output),
            std::move(full_topk_ids),
            std::move(output_chunk_ready_flags),
            output_chunk_size);
    }

    Tensor backward_weight_direct_async_impl(
        Tensor routed_input,
        Tensor grad_output,
        Tensor full_topk_ids,
        std::optional<Tensor> output_chunk_ready_flags,
        int32_t output_chunk_size)
    {
        CHECK_INPUT(routed_input, input_dtype_);
        CHECK_INPUT(grad_output, input_dtype_);
        CHECK_INPUT_LOOSE(full_topk_ids);
        FTRAIN_CHECK(full_topk_ids.scalar_type() == ScalarType::Int ||
                   full_topk_ids.scalar_type() == ScalarType::Long)
            << "full_topk_ids must be int32 or int64";
        FTRAIN_CHECK(input_dtype_ == ScalarType::BFloat16)
            << "AGGroupGemm backward_weight_direct currently supports BF16";
        FTRAIN_CHECK(topk_ == 1)
            << "AGGroupGemm backward_weight_direct currently requires topk=1";
        CHECK_2D(routed_input, full_m_, reduction_dim_);
        CHECK_2D(grad_output, full_m_, output_dim_);
        CHECK_2D(full_topk_ids, full_m_, topk_);

        if (output_chunk_ready_flags.has_value())
        {
            const auto &flags = output_chunk_ready_flags.value();
            FTRAIN_CHECK(flags.is_cuda() && flags.is_contiguous())
                << "output_chunk_ready_flags must be a contiguous HCU tensor";
            FTRAIN_CHECK(flags.scalar_type() == ScalarType::Int)
                << "output_chunk_ready_flags must be int32";
            FTRAIN_CHECK(flags.dim() == 1)
                << "output_chunk_ready_flags must be one-dimensional";
            FTRAIN_CHECK(flags.get_device() == routed_input.get_device())
                << "output_chunk_ready_flags must be on the wgrad device";
            FTRAIN_CHECK(output_chunk_size > 0)
                << "output_chunk_size must be positive";
            FTRAIN_CHECK(
                flags.numel() >=
                (output_dim_ + output_chunk_size - 1) / output_chunk_size)
                << "output_chunk_ready_flags is too small for the FC1 dW "
                   "hidden/output partition";
        }
        else
        {
            FTRAIN_CHECK(output_chunk_size == 0)
                << "output_chunk_size requires output_chunk_ready_flags";
        }

        FTRAIN_CHECK(!wgrad_pending_)
            << "AGGroupGemm backward_weight_direct_async called while a "
               "previous wgrad is still pending; call wait_wgrad() first";

        hipStream_t stream = context_->stream();
        // Inputs are prepared by the Python adapter on the current stream.
        // Hand that dependency to the private wgrad stream explicitly before
        // queueing metadata and GEMM work there.
        CUDA_CHECK(hipEventRecord(wgrad_input_ready_event_, stream));
        CUDA_CHECK(hipStreamWaitEvent(wgrad_stream_,
                                      wgrad_input_ready_event_,
                                      0));
        if (ragged_m_ && rank_m_offsets_dirty_)
        {
            ag_group_gemm_update_rank_m_offsets_impl(
                d_rank_m_offsets_,
                rank_m_offsets_.data(),
                static_cast<int>(rank_m_offsets_.size()),
                wgrad_stream_);
            rank_m_offsets_dirty_ = false;
        }

        // Rebuild metadata from the call's routing IDs.  This keeps the direct
        // wgrad independent of any forward() that may have reused this cached
        // operator for another microbatch.
        ag_group_gemm_prepare_metadata_impl(
            full_topk_ids.data_ptr(),
            sorted_token_ids_.data_ptr<int>(),
            sorted_gather_index_.data_ptr<int>(),
            expert_rank_counts_.data_ptr<int>(),
            expert_rank_counters_.data_ptr<int>(),
            expert_rank_offsets_.data_ptr<int>(),
            global_expert_offsets_.data_ptr<int>(),
            rank_cursors_.data_ptr<int>(),
            rank_tile_info_.data_ptr<int>(),
            tile_experts_.data_ptr<int>(),
            tile_starts_.data_ptr<int>(),
            tile_rank_starts_.data_ptr<int>(),
            tile_rank_ends_.data_ptr<int>(),
            nullptr,
            ntiles_.data_ptr<int>(),
            ragged_m_ ? d_rank_m_offsets_ : nullptr,
            full_m_,
            topk_,
            num_experts_,
            world_size_,
            rank_,
            sorted_capacity_,
            max_tiles_,
            0,
            0,
            false,
            false,
            false,
            false,
            false,
            full_topk_ids.scalar_type(),
            wgrad_stream_);

        auto grad_weight = empty(
            {num_experts_, reduction_dim_, output_dim_},
            routed_input.options());
        if (!wgrad_compact_routed_input_.defined())
        {
            wgrad_compact_routed_input_ = empty(
                {sorted_capacity_, reduction_dim_}, routed_input.options());
            wgrad_compact_grad_output_ = empty(
                {sorted_capacity_, output_dim_}, grad_output.options());
            wgrad_grouped_asm_args_ = empty(
                {num_experts_ * 196},
                routed_input.options().dtype(ScalarType::Byte));
        }

        // The existing kernel computes A^T @ B and stores [E,A_dim,B_dim].
        // Supplying X before dY gives FC1 dW directly as [E,K,N], with each
        // expert consuming only its actual variable-M rows.
        ag_group_gemm_backward_weight_impl(
            routed_input.data_ptr(),
            grad_output.data_ptr(),
            sorted_token_ids_.data_ptr<int>(),
            sorted_gather_index_.data_ptr<int>(),
            global_expert_offsets_.data_ptr<int>(),
            rank_cursors_.data_ptr<int>(),
            wgrad_compact_routed_input_.data_ptr(),
            wgrad_compact_grad_output_.data_ptr(),
            wgrad_grouped_asm_args_.data_ptr(),
            grad_weight.data_ptr(),
            sorted_capacity_,
            full_m_,
            reduction_dim_,
            output_dim_,
            num_experts_,
            topk_,
            world_size_,
            0,
            output_chunk_ready_flags.has_value()
                ? output_chunk_ready_flags.value().data_ptr<int>()
                : nullptr,
            output_chunk_size,
            input_dtype_,
            wgrad_stream_);
        CUDA_CHECK(hipEventRecord(wgrad_done_event_, wgrad_stream_));

        // Keep all raw-pointer inputs alive until the private stream has
        // completed.  In particular, full_topk_ids is an ephemeral tensor
        // constructed by the Python adapter for each backward invocation.
        pending_wgrad_routed_input_ = std::move(routed_input);
        pending_wgrad_grad_output_ = std::move(grad_output);
        pending_wgrad_topk_ids_ = std::move(full_topk_ids);
        pending_wgrad_chunk_ready_flags_ = output_chunk_ready_flags.has_value()
            ? std::move(output_chunk_ready_flags.value())
            : Tensor();
        pending_wgrad_output_ = grad_weight;
        wgrad_pending_ = true;
        return grad_weight;
    }

    void wait_wgrad()
    {
        if (!wgrad_pending_)
        {
            return;
        }
        // Synchronize only the wgrad stream.  MoEReduceRS dgrad/A2A work is
        // independently queued on its own streams and can continue while
        // this event is being waited on by the host.
        CUDA_CHECK(hipEventSynchronize(wgrad_done_event_));
        hipStream_t stream =
            context_->stream();
        CUDA_CHECK(hipStreamWaitEvent(stream, wgrad_done_event_, 0));
        pending_wgrad_routed_input_ = Tensor();
        pending_wgrad_grad_output_ = Tensor();
        pending_wgrad_topk_ids_ = Tensor();
        pending_wgrad_chunk_ready_flags_ = Tensor();
        pending_wgrad_output_ = Tensor();
        wgrad_pending_ = false;
    }

    Tensor gather_input()
    {
        return input_buffer_.slice(0, 0, full_m_);
    }

    Tensor ready_signal_buffer()
    {
        return barrier_buffer_;
    }

    void publish_ready_signal(int32_t slot, int32_t signal_value)
    {
        FTRAIN_CHECK(slot >= 0 && slot < signal_capacity_)
            << "ready signal slot exceeds capacity";
        hipStream_t stream =
            context_->stream();
        ag_group_gemm_publish_signal_release_impl(
            barrier_buffer_.data_ptr<int>() + slot,
            static_cast<uint32_t>(signal_value),
            stream);
    }



    Tensor debug_ntiles()
    {
        return ntiles_.clone();
    }

    Tensor debug_rank_tile_info()
    {
        return rank_tile_info_.clone();
    }

    Tensor debug_rank_tile_problem_info()
    {
        return rank_tile_problem_info_.clone();
    }

    void set_rank_m_sizes(int32_t active_full_m,
                          std::vector<int64_t> rank_m_sizes)
    {
        if (route_spec_.nk_new_a2a_chunk)
        {
            join_a2a_submitter();
            FTRAIN_REQUIRE(supports_ragged_chunk_a2a() ||
                            (active_full_m % world_size_ == 0 &&
                            active_full_m / world_size_ >= 8 &&
                            (rank_m_sizes.empty() ||
                             (rank_m_sizes.size() == static_cast<size_t>(world_size_) &&
                              std::all_of(rank_m_sizes.begin(), rank_m_sizes.end(),
                                  [&](int64_t rows) { return rows == active_full_m / world_size_; })))),
                        route_, " requires equal rank input sizes of at least 8 rows");
        }
        FTRAIN_CHECK(active_full_m > 0) << "active_full_m must be positive";
        FTRAIN_CHECK(active_full_m <= capacity_full_m_)
            << "active_full_m exceeds AGGroupGemm capacity";
        if (rank_m_sizes.empty())
        {
            FTRAIN_CHECK(active_full_m % world_size_ == 0)
                << "active_full_m must be divisible by world_size when "
                   "rank_m_sizes is not provided";
            rank_m_sizes.assign(world_size_, active_full_m / world_size_);
        }
        if (active_full_m == full_m_ &&
            rank_m_sizes_.size() == rank_m_sizes.size() &&
            std::equal(rank_m_sizes_.begin(),
                       rank_m_sizes_.end(),
                       rank_m_sizes.begin(),
                       [](int32_t lhs, int64_t rhs) {
                           return lhs == rhs;
                       }))
        {
            return;
        }
        FTRAIN_CHECK(static_cast<int>(rank_m_sizes.size()) == world_size_)
            << "rank_m_sizes must contain one entry per ETP rank";
        rank_m_sizes_.clear();
        rank_m_offsets_.assign(world_size_ + 1, 0);
        int64_t rank_m_total = 0;
        for (int i = 0; i < world_size_; ++i)
        {
            FTRAIN_CHECK(rank_m_sizes[i] > 0)
                << "rank_m_sizes entries must be positive";
            FTRAIN_CHECK(rank_m_sizes[i] <= std::numeric_limits<int32_t>::max())
                << "rank_m_sizes entry exceeds int32 range";
            rank_m_sizes_.push_back(static_cast<int32_t>(rank_m_sizes[i]));
            rank_m_total += rank_m_sizes[i];
            FTRAIN_CHECK(rank_m_total <= active_full_m)
                << "rank_m_sizes prefix exceeds active_full_m";
            rank_m_offsets_[i + 1] = static_cast<int32_t>(rank_m_total);
        }
        FTRAIN_CHECK(rank_m_offsets_.back() == active_full_m)
            << "sum(rank_m_sizes) must equal active_full_m";
        full_m_ = active_full_m;
        local_m_ = rank_m_sizes_[rank_];
        // Force an explicit route to retain ragged offsets even when all
        // owner sizes are equal.  This is used only by the isolated A/B
        // route to separate scheduler cost from input-size imbalance.
        ragged_m_ =
            native_kn_ragged_ring_ ||
            get_bool_from_env("FTRAIN_AG_GROUP_GEMM_FORCE_RAGGED_M", false) ||
            std::any_of(
                rank_m_sizes_.begin() + 1,
                rank_m_sizes_.end(),
                [&](int32_t value) { return value != rank_m_sizes_.front(); });
        local_m_offset_ = rank_m_offsets_[rank_];
        rank_m_offsets_dirty_ = ragged_m_;
        metadata_cache_valid_ = false;
    }

    Tensor n_chunk_done_flags()
    {
        return n_chunk_done_flags_;
    }

private:
    std::shared_ptr<Context> context_;
    int32_t nnodes_;
    int32_t full_m_;
    int32_t capacity_full_m_;
    // Semantic names for the native ABI fields: output N then reduction K.
    int32_t output_dim_;
    int32_t reduction_dim_;
    int32_t num_experts_;
    int32_t topk_;
    ScalarType input_dtype_;
    ScalarType output_dtype_;
    bool allocate_output_on_init_;
    int world_size_;
    int rank_;
    int local_world_size_;
    int local_rank_;
    std::string route_;
    std::function<void()> a2a_deferred_submit_;
    std::thread a2a_submit_thread_;
    std::exception_ptr a2a_submit_error_;
    AGRouteSpec route_spec_;
    bool enk_ring_chunk_ready_ = false;
    std::function<void()> enk_chunk_transport_;
    std::function<void()> kn_chunk_transport_;
    bool native_kn_ragged_ring_ = false;
    int32_t local_m_ = 0;
    int32_t local_m_offset_ = 0;
    bool ragged_m_ = false;
    std::vector<int32_t> rank_m_sizes_;
    std::vector<int32_t> rank_m_offsets_;
    int32_t *d_rank_m_offsets_ = nullptr;
    bool rank_m_offsets_dirty_ = false;
    int signal_capacity_ = std::max(
        MAX_NUM_SIGNAL,
        std::min(get_int_from_env_local(
                     "FTRAIN_AG_GROUP_GEMM_SIGNAL_CAPACITY",
                     MAX_NUM_SIGNAL),
                 4096));

    std::vector<Tensor> input_buffers_;
    Tensor input_buffer_;
    void **d_input_ptrs_ = nullptr;
    Tensor output_buffer_;
    Tensor sorted_token_ids_;
    Tensor sorted_gather_index_;
    Tensor expert_rank_counts_;
    Tensor expert_rank_counters_;
    Tensor expert_rank_offsets_;
    Tensor global_expert_offsets_;
    Tensor rank_cursors_;
    Tensor rank_tile_info_;
    Tensor rank_tile_problem_info_;
    Tensor tile_experts_;
    Tensor tile_starts_;
    Tensor tile_rank_starts_;
    Tensor tile_rank_ends_;
    Tensor rank_tile_ready_flags_;
    Tensor ntiles_;
    Tensor n_chunk_done_counts_;
    Tensor n_chunk_done_flags_;
    Tensor packed_weight_;
    std::vector<Tensor> barrier_buffers_;
    Tensor barrier_buffer_;
    Tensor signal_one_;
    Tensor signal_value_tensor_;
    int32_t sorted_capacity_ = 0;
    int32_t aggregate_max_tiles_ = 0;
    int32_t max_tiles_ = 0;
    int32_t n_chunk_done_groups_ = 0;
    int32_t group_n_chunk_size_ = 256;
    const void *packed_weight_src_ = nullptr;
    int64_t packed_weight_version_ = -1;
    Tensor metadata_topk_tensor_;
    const void *metadata_topk_src_ = nullptr;
    int64_t metadata_topk_version_ = -1;
    ScalarType metadata_topk_dtype_ = ScalarType::Undefined;
    bool metadata_strict_rank_tile_ = false;
    bool metadata_split_local_rank_ = false;
    bool metadata_isolate_local_owner_ = false;
    bool metadata_build_problem_info_ = false;
    bool metadata_stage_signal_layout_ = false;
    bool metadata_stage_signal_zigzag_ = false;
    int32_t metadata_signal_chunk_rows_ = 0;
    int32_t metadata_chunks_per_rank_ = 0;
    bool metadata_cache_valid_ = false;
    bool last_forward_valid_ = false;
    bool last_forward_used_rank_signal_ = false;
    int last_signal_value_ = 0;
    int signal_value_ = 0;
    bool logged_route_ = false;
    int profile_forward_count_ = 0;
    bool comm_stream_high_priority_ = false;
    hipStream_t comm_stream_ = nullptr;
    AGWeightPackStream weight_pack_stream_;
    hipStream_t wgrad_stream_ = nullptr;
    std::vector<hipStream_t> push_streams_;
    int32_t ring_channels_ = 1;
    std::vector<std::vector<hipStream_t>> ring_extra_streams_;
    std::vector<std::vector<hipEvent_t>> ring_channel_done_events_;
    hipEvent_t weight_pack_start_event_ = nullptr;
    hipEvent_t weight_pack_done_event_ = nullptr;
    hipEvent_t wgrad_metadata_ready_event_ = nullptr;
    hipEvent_t wgrad_done_event_ = nullptr;
    hipEvent_t wgrad_input_ready_event_ = nullptr;
    Tensor pending_wgrad_routed_input_;
    Tensor pending_wgrad_grad_output_;
    Tensor pending_wgrad_topk_ids_;
    Tensor pending_wgrad_chunk_ready_flags_;
    Tensor pending_wgrad_output_;
    Tensor wgrad_compact_routed_input_;
    Tensor wgrad_compact_grad_output_;
    Tensor wgrad_grouped_asm_args_;
    bool wgrad_pending_ = false;
    hipEvent_t ready_event_ = nullptr;
    hipEvent_t remote_ready_event_ = nullptr;
    hipEvent_t a2a_consumer_done_event_ = nullptr;
    hipEvent_t chunk_bootstrap_event_ = nullptr;
    std::vector<hipEvent_t> push_done_events_;
#ifndef FTRAIN_SHM_USE_NVSHMEM
    std::vector<Tensor> sync_buffers_;
#endif
};

AGGroupGemm::AGGroupGemm(std::shared_ptr<Context> context,
                         int32_t nnodes,
                         int32_t full_m,
                         int32_t n_dim,
                         int32_t k_dim,
                         int32_t num_experts,
                         int32_t topk,
                         ScalarType input_dtype,
                         ScalarType output_dtype,
                         bool allocate_output_on_init)
    : AGGroupGemm(std::move(context),
                  nnodes,
                  full_m,
                  n_dim,
                  k_dim,
                  num_experts,
                  topk,
                  input_dtype,
                  output_dtype,
                  allocate_output_on_init,
                  {})
{
}

AGGroupGemm::AGGroupGemm(std::shared_ptr<Context> context,
                         int32_t nnodes,
                         int32_t full_m,
                         int32_t n_dim,
                         int32_t k_dim,
                         int32_t num_experts,
                         int32_t topk,
                         ScalarType input_dtype,
                         ScalarType output_dtype,
                         bool allocate_output_on_init,
                         std::vector<int64_t> rank_m_sizes)
    : AGGroupGemm(std::move(context),
                  nnodes,
                  full_m,
                  n_dim,
                  k_dim,
                  num_experts,
                  topk,
                  input_dtype,
                  output_dtype,
                  allocate_output_on_init,
                  std::move(rank_m_sizes),
                  -1)
{
}

AGGroupGemm::AGGroupGemm(std::shared_ptr<Context> context,
                         int32_t nnodes,
                         int32_t full_m,
                         int32_t n_dim,
                         int32_t k_dim,
                         int32_t num_experts,
                         int32_t topk,
                         ScalarType input_dtype,
                         ScalarType output_dtype,
                         bool allocate_output_on_init,
                         std::vector<int64_t> rank_m_sizes,
                         int32_t capacity_full_m,
                         const std::string &route)
    : pimpl_(std::make_unique<AGGroupGemmImpl>(std::move(context),
                                               nnodes,
                                               full_m,
                                               n_dim,
                                               k_dim,
                                               num_experts,
                                               topk,
                                               input_dtype,
                                               output_dtype,
                                               allocate_output_on_init,
                                               std::move(rank_m_sizes),
                                               capacity_full_m,
                                               route))
{
}

void AGGroupGemm::set_rank_m_sizes(int32_t active_full_m,
                                   std::vector<int64_t> rank_m_sizes)
{
    pimpl_->set_rank_m_sizes(active_full_m, std::move(rank_m_sizes));
}

AGGroupGemm::~AGGroupGemm() = default;

Tensor AGGroupGemm::forward(Tensor input,
                                   Tensor weight,
                                   Tensor full_topk_ids,
                                   std::optional<Tensor> output)
{
    return pimpl_->forward(std::move(input),
                           std::move(weight),
                           std::move(full_topk_ids),
                           std::move(output));
}

Tensor AGGroupGemm::backward_weight(Tensor full_routed_input)
{
    return pimpl_->backward_weight(std::move(full_routed_input));
}

Tensor AGGroupGemm::backward_weight_async(Tensor full_routed_input)
{
    return pimpl_->backward_weight_async(std::move(full_routed_input));
}

Tensor AGGroupGemm::backward_weight_direct(
    Tensor routed_input,
    Tensor grad_output,
    Tensor full_topk_ids)
{
    return pimpl_->backward_weight_direct(std::move(routed_input),
                                          std::move(grad_output),
                                          std::move(full_topk_ids));
}

Tensor AGGroupGemm::backward_weight_direct_async(
    Tensor routed_input,
    Tensor grad_output,
    Tensor full_topk_ids)
{
    return pimpl_->backward_weight_direct_async(std::move(routed_input),
                                                std::move(grad_output),
                                                std::move(full_topk_ids));
}

Tensor AGGroupGemm::backward_weight_direct_chunked_async(
    Tensor routed_input,
    Tensor grad_output,
    Tensor full_topk_ids,
    Tensor output_chunk_ready_flags,
    int32_t output_chunk_size)
{
    return pimpl_->backward_weight_direct_chunked_async(
        std::move(routed_input),
        std::move(grad_output),
        std::move(full_topk_ids),
        std::move(output_chunk_ready_flags),
        output_chunk_size);
}

void AGGroupGemm::wait_wgrad()
{
    pimpl_->wait_wgrad();
}

Tensor AGGroupGemm::gather_input()
{
    return pimpl_->gather_input();
}

Tensor AGGroupGemm::ready_signal_buffer()
{
    return pimpl_->ready_signal_buffer();
}

void AGGroupGemm::publish_ready_signal(int32_t slot, int32_t signal_value)
{
    pimpl_->publish_ready_signal(slot, signal_value);
}



Tensor AGGroupGemm::debug_ntiles()
{
    return pimpl_->debug_ntiles();
}

Tensor AGGroupGemm::debug_rank_tile_info()
{
    return pimpl_->debug_rank_tile_info();
}

Tensor AGGroupGemm::debug_rank_tile_problem_info()
{
    return pimpl_->debug_rank_tile_problem_info();
}

Tensor AGGroupGemm::n_chunk_done_flags()
{
    return pimpl_->n_chunk_done_flags();
}

}
} // namespace moe
} // namespace ftrain
