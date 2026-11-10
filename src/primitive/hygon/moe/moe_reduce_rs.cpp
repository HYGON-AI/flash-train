#include "moe_core.h"
#include "hip/hip_runtime.h"
#include "hip_kernels/ag_group_gemm_kernels.h"
#include "hip_kernels/moe_fc2_gemm.h"
#include "hip_kernels/moe_reduce_rs_kernels.h"
#include "moe_core/moe_reduce_rs.h"
#include "moe_core/runtime_utils.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace ftrain
{
namespace moe
{
namespace core
{

namespace
{

int ceil_div(int value, int divisor)
{
    FTRAIN_CHECK(divisor > 0);
    return (value + divisor - 1) / divisor;
}

int align_up(int value, int alignment)
{
    FTRAIN_CHECK(alignment > 0);
    return ceil_div(value, alignment) * alignment;
}

int get_int_from_env(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr)
    {
        return default_value;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end == value ? default_value : static_cast<int>(parsed);
}

std::string normalize_gemm_variant(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return std::tolower(character); });
    if (value.empty() || value == "auto")
    {
        return "rs_nn";
    }
    FTRAIN_CHECK(value == "rs_nn" || value == "rs_nt")
        << "Unsupported MoEReduceRS grouped-GEMM variant: " << value
        << "; supported variants: rs_nn, rs_nt";
    return value;
}

class ScopedEnvironmentValues
{
public:
    ScopedEnvironmentValues(
        std::initializer_list<std::pair<const char *, std::string>> values)
    {
        old_values_.reserve(values.size());
        for (const auto &[name, value] : values)
        {
            const char *old = std::getenv(name);
            old_values_.emplace_back(
                name, old != nullptr ? std::optional<std::string>(old) : std::nullopt);
            setenv(name, value.c_str(), 1);
        }
    }

    ~ScopedEnvironmentValues()
    {
        for (auto it = old_values_.rbegin(); it != old_values_.rend(); ++it)
        {
            if (it->second)
            {
                setenv(it->first, it->second->c_str(), 1);
            }
            else
            {
                unsetenv(it->first);
            }
        }
    }

    ScopedEnvironmentValues(const ScopedEnvironmentValues &) = delete;
    ScopedEnvironmentValues &operator=(const ScopedEnvironmentValues &) = delete;

private:
    std::vector<std::pair<const char *, std::optional<std::string>>> old_values_;
};

ScopedEnvironmentValues make_gemm_contract_environment()
{
    return {{
        {"FTRAIN_AG_GROUP_GEMM_ASM_MT256", "1"},
        {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT", "1"},
        {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", "1"},
        {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_WGM8", "1"},
        {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", "0"},
        {"FTRAIN_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_N_CHUNK_SIGNAL", "1"},
        {"FTRAIN_AG_GROUP_GEMM_RANK_TILE_ENABLE", "1"},
    }};
}

void check_supported_dtype(ScalarType dtype)
{
    FTRAIN_CHECK(dtype == ScalarType::BFloat16)
        << "MoEReduceRS FC2 GEMM only supports BF16";
}

void all_ranks_sync(hipStream_t stream,
                    std::optional<std::vector<Tensor>> sync_buffers,
                    std::optional<int> rank)
{
    if (!get_bool_from_env("FTRAIN_BARRIER_ALL_ON_STREAM", true))
    {
        return;
    }
#ifdef FTRAIN_SHM_USE_NVSHMEM
    barrier_all_on_stream(stream);
#else
    barrier_all_on_stream(stream, std::move(sync_buffers), rank);
#endif
}

struct ChunkConfig
{
    int chunk_size_n;
    int num_groups;
    bool n_chunk_signal;
};

enum class MoERSRouteKind
{
    Auto,
    A2A,
    A2ARaggedSignal,
};

MoERSRouteKind resolve_moe_rs_route(const std::string &route)
{
    std::string value = route;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (value.empty() || value == "auto")
    {
        return MoERSRouteKind::Auto;
    }
    if (value == "a2a")
    {
        return MoERSRouteKind::A2A;
    }
    if (value == "a2a_ragged_signal")
    {
        return MoERSRouteKind::A2ARaggedSignal;
    }
    FTRAIN_CHECK(false)
        << "Unsupported MoEReduceRS route: " << route
        << " (supported: auto, a2a, a2a_ragged_signal)";
    return MoERSRouteKind::Auto;
}

bool moe_route_owns_a2a(MoERSRouteKind kind)
{
    return kind == MoERSRouteKind::A2A ||
        kind == MoERSRouteKind::A2ARaggedSignal;
}

int get_rank_tile_block_m(ScalarType dtype)
{
    const int default_block_m =
        dtype == ScalarType::BFloat16 ? 128 : 32;
    const int requested = get_int_from_env(
        "FTRAIN_MOE_RS_CPP_GEMM_RANK_TILE_BM", default_block_m);
    return requested > 0 && requested % 16 == 0 ? requested : default_block_m;
}

ChunkConfig make_chunk_config(int n_dim,
                              int n_chunks,
                              bool use_rank_tile,
                              ScalarType dtype,
                              int k_dim,
                              bool route_owned)
{
    FTRAIN_CHECK(n_dim > 0);

    // The old host allowed N-chunk hipBLAS only behind an experimental flag.
    // hipBLAS is intentionally not reconstructed, but retaining this gate here
    // reproduces the old ChunkConfig contract for the rank-tile route.
    if (!use_rank_tile &&
        !get_bool_from_env("FTRAIN_MOE_RS_ENABLE_N_CHUNK_GEMM", false))
    {
        return {n_dim, 1, false};
    }

    int requested = n_chunks;
    if (requested <= 0)
    {
        requested = get_int_from_env("FTRAIN_MOE_RS_N_CHUNKS", 0);
    }

    const bool auto_n_chunk =
        get_bool_from_env("FTRAIN_MOE_RS_HCU_AUTO_N_CHUNK", true);
    if (requested <= 0)
    {
        const bool auto_shape_supported =
            auto_n_chunk && use_rank_tile &&
            dtype == ScalarType::BFloat16 && n_dim % 256 == 0 &&
            k_dim % 32 == 0 && get_rank_tile_block_m(dtype) == 128;
        if (auto_shape_supported)
        {
            const int default_a2a_target = n_dim == 4096
                ? 1024
                : n_dim == 8192
                ? 512
                : n_dim > 8192 ? 1024 : 256;
            const int default_target = default_a2a_target;
            int target = get_int_from_env(
                "FTRAIN_MOE_RS_HCU_AUTO_N_CHUNK_TARGET", default_target);
            target = get_int_from_env(
                "FTRAIN_MOE_RS_HCU_AUTO_N_CHUNK_A2A_TARGET", target);
            target = std::max(256, target);

            const int default_max_groups = (n_dim > 8192 ? 64 : 32);
            int max_groups = get_int_from_env(
                "FTRAIN_MOE_RS_HCU_AUTO_N_CHUNKS_MAX", default_max_groups);
            max_groups = get_int_from_env(
                "FTRAIN_MOE_RS_HCU_AUTO_N_CHUNKS_A2A_MAX", max_groups);
            max_groups = std::max(1, max_groups);

            requested = std::min(max_groups, ceil_div(n_dim, target));
            requested = std::max(1, requested);
            // The backup host only imposed the four-group floor for the
            // large-N Ring schedule.  Applying it to all auto-selected
            // shapes changes N=4096 Ring from the original 2 x 2048 layout
            // to 4 x 1024 and no longer matches its signal contract.
        }
        else
        {
            requested = 1;
        }
    }

    requested = std::max(1, requested);
    int chunk_size_n = n_dim;
    if (requested >= 2)
    {
        const int alignment =
            dtype == ScalarType::BFloat16 ? 256 : 128;
        chunk_size_n = std::min(
            n_dim, align_up(ceil_div(n_dim, requested), alignment));
    }
    const int num_groups = ceil_div(n_dim, chunk_size_n);

    const bool signal_default =
        requested >= 2 || auto_n_chunk ||
        get_int_from_env("FTRAIN_MOE_RS_N_CHUNKS", 0) >= 2;
    const bool n_chunk_signal =
        use_rank_tile && dtype == ScalarType::BFloat16 &&
        num_groups >= 2 && n_dim % 256 == 0 && chunk_size_n % 256 == 0 &&
        k_dim % 32 == 0 && get_rank_tile_block_m(dtype) == 128 &&
        get_bool_from_env("FTRAIN_MOE_RS_AG_RANK_TILE_N_CHUNK_SIGNAL",
                          signal_default || route_owned);
    return {chunk_size_n, num_groups, n_chunk_signal};
}

} // namespace

class MoEReduceRS::MoEReduceRSImpl
{
public:
    MoEReduceRSImpl(std::shared_ptr<Context> context,
                    int32_t nnodes,
                    int32_t max_ntokens,
                    int32_t n_dim,
                    int32_t k_dim,
                    int32_t num_experts,
                    int32_t topk,
                    ScalarType input_dtype,
                    int32_t n_chunks,
                    std::string route,
                    std::string gemm_variant)
        : route_kind_(resolve_moe_rs_route(route)), route_(std::move(route)),
          gemm_variant_(normalize_gemm_variant(std::move(gemm_variant))),
          context_(std::move(context)), nnodes_(nnodes),
          max_ntokens_(max_ntokens), n_dim_(n_dim), k_dim_(k_dim),
          num_experts_(num_experts), topk_(topk), input_dtype_(input_dtype),
          n_chunks_(n_chunks), world_size_(context_->world_size()),
          rank_(context_->rank()),
          local_world_size_(nnodes_ > 0 ? world_size_ / nnodes_ : 0),
          local_rank_(local_world_size_ > 0 ? rank_ % local_world_size_ : 0),
          block_m_(get_rank_tile_block_m(input_dtype)),
          max_rank_tiles_(ceil_div(max_ntokens * topk, block_m_) + num_experts),
          source_sorted_capacity_(max_ntokens * topk + num_experts * 256),
          a2a_chunks_(make_chunk_config(
              n_dim, n_chunks, true, input_dtype, k_dim,
              moe_route_owns_a2a(route_kind_))),
          reduce_buffer_n_(a2a_chunks_.chunk_size_n),
          max_num_groups_(a2a_chunks_.num_groups),
          a2a_split_pipeline_enabled_(
              nnodes_ == 1 && world_size_ >= 2 &&
              get_bool_from_env("FTRAIN_MOE_RS_A2A_SPLIT_PIPELINE", false)),
          // Hardware wait is the low-latency route through 8192.  For a
          // sufficiently large M=O(16K) wide-GEMM, it also wins through a
          // 16K output width when the logical N groups remain coarse (at
          // least 1024 columns, no more than 16 groups).  Finer groups turn
          // the scatter producer into a persistent-GEMM competitor, so keep
          // those on the ordinary scheduled route.  The environment variable
          // remains an explicit override for experiments.
          a2a_hardware_wait_enabled_(
              nnodes_ == 1 && world_size_ >= 2 &&
              (moe_route_owns_a2a(route_kind_)
                   ? (n_dim_ <= 8192 ||
                      (n_dim_ <= 16384 &&
                       max_ntokens_ >= 16384 &&
                       a2a_chunks_.chunk_size_n >= 1024 &&
                       a2a_chunks_.num_groups <= 16))
                   : get_bool_from_env(
                       "FTRAIN_MOE_RS_A2A_HARDWARE_WAIT",
                       n_dim_ <= 8192 ||
                           (n_dim_ <= 16384 &&
                            max_ntokens_ >= 16384 &&
                            a2a_chunks_.chunk_size_n >= 1024 &&
                            a2a_chunks_.num_groups <= 16)))),
          a2a_signal_pipeline_enabled_(
              nnodes_ == 1 && world_size_ >= 2 &&
              (moe_route_owns_a2a(route_kind_) ||
               get_bool_from_env("FTRAIN_MOE_RS_A2A_SIGNAL_PIPELINE", false) ||
               get_bool_from_env("FTRAIN_MOE_RS_A2A_RAGGED_SIGNAL", true) ||
               a2a_split_pipeline_enabled_ ||
               a2a_hardware_wait_enabled_)),
          a2a_signal_stream_wait_enabled_(
              a2a_signal_pipeline_enabled_ &&
              get_bool_from_env("FTRAIN_MOE_RS_A2A_SIGNAL_STREAM_WAIT", true)),
          a2a_wave64_512_scatter_enabled_(
              a2a_hardware_wait_enabled_ &&
              get_bool_from_env(
                  "FTRAIN_MOE_RS_A2A_WAVE64_512_SCATTER",
                  get_bool_from_env("FTRAIN_MOE_RS_A2A_WAVE64_512", true))),
          a2a_wave64_512_sum_enabled_(
              a2a_hardware_wait_enabled_ &&
              get_bool_from_env(
                  "FTRAIN_MOE_RS_A2A_WAVE64_512_SUM",
                  get_bool_from_env("FTRAIN_MOE_RS_A2A_WAVE64_512", true))),
          a2a_signal_block_budget_(std::max(
              1,
              std::min(kA2AResidentBlocks,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_SIGNAL_BLOCK_BUDGET", 16)))),
          a2a_signal_producer_block_budget_(std::max(
              1,
              std::min(kA2AResidentBlocks,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_PRODUCER_BLOCK_BUDGET",
                           a2a_wave64_512_scatter_enabled_ &&
                                   world_size_ == 8 && topk_ == 6 &&
                                   a2a_chunks_.chunk_size_n == 512 &&
                                   n_dim_ % 512 == 0
                               ? 32
                               : 24)))),
          a2a_signal_consumer_block_budget_(std::max(
              1,
              std::min(kA2AResidentBlocks,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_CONSUMER_BLOCK_BUDGET",
                           a2a_hardware_wait_enabled_ ? 24 : 8)))),
          a2a_staged_block_budget_enabled_(
              a2a_hardware_wait_enabled_ &&
              get_bool_from_env("FTRAIN_MOE_RS_A2A_STAGED_BLOCK_BUDGET", false)),
          a2a_signal_early_producer_block_budget_(std::max(
              1,
              std::min(kA2AResidentBlocks,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_EARLY_PRODUCER_BLOCK_BUDGET",
                           n_dim_ > 8192 && n_dim_ <= 16384 &&
                                   max_ntokens_ >= 16384 &&
                                   a2a_chunks_.chunk_size_n >= 1024 &&
                                   a2a_chunks_.num_groups <= 16
                               ? 8
                               : 16)))),
          a2a_signal_early_consumer_block_budget_(std::max(
              1,
              std::min(kA2AResidentBlocks,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_EARLY_CONSUMER_BLOCK_BUDGET",
                           n_dim_ > 8192 && n_dim_ <= 16384 &&
                                   max_ntokens_ >= 16384 &&
                                   a2a_chunks_.chunk_size_n >= 1024 &&
                                   a2a_chunks_.num_groups <= 16
                               ? 4
                               : 8)))),
          a2a_signal_full_budget_start_group_(std::max(
              0,
              std::min(
                  a2a_chunks_.num_groups,
                  get_int_from_env(
                      "FTRAIN_MOE_RS_A2A_FULL_BUDGET_START_GROUP",
                      (a2a_chunks_.num_groups * 3) / 4)))),
          a2a_cu_partition_enabled_(
              a2a_hardware_wait_enabled_ &&
              get_bool_from_env("FTRAIN_MOE_RS_A2A_CU_PARTITION", false)),
          a2a_comm_cu_count_(std::max(
              1,
              get_int_from_env("FTRAIN_MOE_RS_A2A_COMM_CU_COUNT", 16))),
          a2a_group_pipeline_enabled_(
              get_bool_from_env(
                  "FTRAIN_MOE_RS_A2A_GROUP_PIPELINE",
                  false)),
          a2a_paced_scatter_enabled_(
              nnodes_ == 1 && world_size_ >= 2 &&
              get_bool_from_env("FTRAIN_MOE_RS_A2A_PACED_SCATTER", false)),
          a2a_paced_scatter_check_iterations_(std::max(
              1,
              std::min(64,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_PACED_SCATTER_CHECK_ITERS", 4)))),
          a2a_paced_scatter_sleep_steps_(std::max(
              0,
              std::min(64,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_PACED_SCATTER_SLEEP_STEPS", 8)))),
          a2a_admitted_scatter_enabled_(
              nnodes_ == 1 && world_size_ >= 2 &&
              get_bool_from_env("FTRAIN_MOE_RS_A2A_ADMITTED_SCATTER", false)),
          a2a_admitted_scatter_low_priority_(get_bool_from_env(
              "FTRAIN_MOE_RS_A2A_ADMITTED_SCATTER_LOW_PRIORITY", true)),
          a2a_admitted_scatter_batch_groups_(std::max(
              1,
              std::min(64,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_ADMITTED_SCATTER_BATCH_GROUPS",
                           1)))),
          a2a_admitted_scatter_early_blocks_(std::max(
              1,
              std::min(256,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_ADMITTED_SCATTER_EARLY_BLOCKS",
                           16)))),
          a2a_admitted_scatter_final_blocks_(std::max(
              1,
              std::min(256,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_ADMITTED_SCATTER_FINAL_BLOCKS",
                           128)))),
          a2a_adaptive_admission_enabled_(
              nnodes_ == 1 && world_size_ >= 2 &&
              get_bool_from_env("FTRAIN_MOE_RS_A2A_ADAPTIVE_ADMISSION", false)),
          a2a_adaptive_admission_low_priority_(get_bool_from_env(
              "FTRAIN_MOE_RS_A2A_ADAPTIVE_ADMISSION_LOW_PRIORITY", true)),
          a2a_adaptive_admission_lead_groups_(std::max(
              1,
              std::min(64,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_ADAPTIVE_ADMISSION_LEAD_GROUPS",
                           1)))),
          a2a_adaptive_admission_lead_percent_(std::max(
              1,
              std::min(100,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_ADAPTIVE_ADMISSION_LEAD_PERCENT",
                           50)))),
          a2a_adaptive_admission_early_blocks_(std::max(
              1,
              std::min(256,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_ADAPTIVE_ADMISSION_EARLY_BLOCKS",
                           16)))),
          a2a_adaptive_admission_final_blocks_(std::max(
              1,
              std::min(256,
                       get_int_from_env(
                           "FTRAIN_MOE_RS_A2A_ADAPTIVE_ADMISSION_FINAL_BLOCKS",
                           128)))),
          a2a_decoupled_scatter_enabled_(false)
    {
        FTRAIN_CHECK(arch_check()) << "Unsupported arch: " << get_device_name();
        FTRAIN_CHECK(nnodes_ > 0 && world_size_ > 0 &&
                   world_size_ % nnodes_ == 0)
            << "invalid nnodes/world_size";
        FTRAIN_CHECK(local_world_size_ > 0);
        FTRAIN_CHECK(max_ntokens_ > 0);
        FTRAIN_CHECK(n_dim_ > 0 && k_dim_ > 0 && num_experts_ > 0 && topk_ > 0);
        check_supported_dtype(input_dtype_);
        // The flat-input grouped-GEMM and reduction kernels consume `topk`
        // as runtime metadata.  Mixtral FC1 backward uses one routed expert
        // per row, so accepting topk=1 is required; keep the positive check
        // above as the contract for other supported routing arities.

        // The device-address wait capability is what the hardware-wait
        // kernels actually need, so resolve it once for every single-node
        // multi-rank instance instead of only when the fixed-M shape policy
        // happens to request it.  The ragged signal route has no stream-wait
        // alternative and therefore keys off the capability, while the
        // fixed-M route keeps its measured shape policy below.
        int can_wait_value = 0;
        if (nnodes_ == 1 && world_size_ >= 2)
        {
            CUDA_CHECK(hipDeviceGetAttribute(
                &can_wait_value,
                hipDeviceAttributeCanUseStreamWaitValue,
                context_->device()));
        }
        a2a_hardware_wait_capable_ = can_wait_value != 0;
        if (a2a_signal_stream_wait_enabled_ || a2a_hardware_wait_enabled_ ||
            a2a_admitted_scatter_enabled_ ||
            a2a_adaptive_admission_enabled_)
        {
            a2a_signal_stream_wait_enabled_ =
                a2a_signal_stream_wait_enabled_ && can_wait_value != 0;
            a2a_admitted_scatter_enabled_ =
                a2a_admitted_scatter_enabled_ && can_wait_value != 0;
            a2a_adaptive_admission_enabled_ =
                a2a_adaptive_admission_enabled_ && can_wait_value != 0;
            if (!can_wait_value)
            {
                // The hardware-wait route uses the same HIP stream wait-value
                // capability for both AG done flags and cross-rank ready
                // epochs. Fall back to the existing signal route when this
                // DTK/runtime cannot schedule a device-address wait.
                a2a_hardware_wait_enabled_ = false;
            }
        }
        // The reconstructed MT256 AG family (v4/v5 direct-input and v7/v10/
        // v12/v13/v14/v15/v36 direct-output family has a
        // 64-element direct-input K contract. The launcher applies the
        // variant-specific direct-output limits. Reject common unsupported
        // layouts before RS state has been published.
        FTRAIN_CHECK(n_dim_ % 256 == 0)
            << "Recovered BF16 AG rank-tile MoEReduceRS requires n_dim "
               "to be a multiple of 256, got "
            << n_dim_;
        FTRAIN_CHECK(k_dim_ % 64 == 0)
            << "Source MT256 BF16 AG rank-tile MoEReduceRS requires k_dim "
               "to be a multiple of 64, got "
            << k_dim_;

        const int device = context_->device();
#ifndef FTRAIN_SHM_USE_NVSHMEM
        if (nnodes_ == 1 && world_size_ >= 2 &&
            get_bool_from_env("FTRAIN_MOE_RS_A2A_COOPERATIVE_FUSED",
                              n_dim_ <= 8192))
        {
            hipDeviceProp_t properties{};
            CUDA_CHECK(hipGetDeviceProperties(&properties, device));
            const bool request_hip_cooperative_launch = get_bool_from_env(
                "FTRAIN_MOE_RS_A2A_HIP_COOPERATIVE_LAUNCH", true);
            if (request_hip_cooperative_launch)
            {
                int cooperative_supported = 0;
                CUDA_CHECK(hipDeviceGetAttribute(
                    &cooperative_supported,
                    hipDeviceAttributeCooperativeLaunch,
                    device));
                a2a_hip_cooperative_launch_enabled_ =
                    cooperative_supported != 0;
            }
            // The software fallback still requires all 32 blocks to be
            // resident.  The formal cooperative path performs a per-kernel
            // occupancy check immediately before launch.
            a2a_cooperative_fused_enabled_ =
                a2a_hip_cooperative_launch_enabled_ ||
                properties.multiProcessorCount >= kA2AResidentBlocks;
        }
#endif
        auto device_options = context_->tensor_options().device_index(device);
        auto int_options = device_options.dtype(ScalarType::Int);

        gemm_output_ = empty(
            {static_cast<int64_t>(max_ntokens_) * topk_, n_dim_},
            device_options.dtype(input_dtype_));
        topk_stage_ = empty({max_ntokens_, n_dim_},
                                   device_options.dtype(ScalarType::Float));
        output_buffer_ = empty({std::max(1, max_ntokens_ / world_size_), n_dim_},
                                      device_options.dtype(input_dtype_));

        expert_counts_ = empty({num_experts_}, int_options);
        expert_offsets_ = empty({num_experts_ + 1}, int_options);
        expert_rank_bounds_ = empty({num_experts_ * 2}, int_options);
        rank_tile_info_ = empty({max_rank_tiles_ * 5}, int_options);
        metadata_scratch0_ = empty({max_rank_tiles_}, int_options);
        sorted_index_workspace_ = empty(
            {static_cast<int64_t>(max_ntokens_) * topk_}, int_options);

        source_sorted_token_ids_ =
            empty({source_sorted_capacity_}, int_options);
        source_sorted_gather_index_ =
            empty({source_sorted_capacity_}, int_options);
        source_expert_rank_counts_ = empty({num_experts_}, int_options);
        source_expert_rank_counters_ = empty({num_experts_}, int_options);
        source_expert_rank_offsets_ =
            empty({num_experts_ + 1}, int_options);
        source_global_expert_offsets_ =
            empty({num_experts_ + 1}, int_options);
        source_rank_cursors_ = empty({num_experts_ * 2}, int_options);
        source_rank_tile_info_ =
            empty({max_rank_tiles_, 5}, int_options);
        source_tile_experts_ = empty({max_rank_tiles_}, int_options);
        source_tile_starts_ = empty({max_rank_tiles_}, int_options);
        source_tile_rank_starts_ =
            empty({max_rank_tiles_}, int_options);
        source_tile_rank_ends_ = empty({max_rank_tiles_}, int_options);
        source_ntiles_ = empty({1}, int_options);
        CUDA_CHECK(hipMalloc(&d_rank_m_offsets_,
                             static_cast<size_t>(world_size_ + 1) *
                                 sizeof(int)));

        n_chunk_done_counts_ = zeros({max_num_groups_}, int_options);
        n_chunk_done_flags_ = zeros({max_num_groups_}, int_options);
        gemm_done_counts_ = zeros({max_num_groups_}, int_options);
        gemm_done_flags_ = zeros({max_num_groups_}, int_options);
        if (a2a_adaptive_admission_enabled_)
        {
            const int physical_n_tiles = ceil_div(n_dim_, 256);
            const int64_t control_words =
                static_cast<int64_t>(max_rank_tiles_) *
                    physical_n_tiles +
                max_num_groups_;
            FTRAIN_CHECK(control_words > 0 &&
                       control_words <= std::numeric_limits<int>::max());
            v36_task_control_capacity_ = static_cast<int>(control_words);
            v36_task_control_ = zeros({control_words}, int_options);
        }

        // Chunked ProcessGroup reduce-scatter cannot consume an N-strided
        // view directly.  Store its inputs and outputs chunk-major so every
        // collective receives a compact [tokens, chunk_n] tensor.  The total
        // storage is still one full input plus one full output, independent
        // of the number of groups.
        if (nnodes_ == 1)
        {
            // The recovered A2A/Ring kernels dereference every peer through
            // HIP IPC (or NVSHMEM).  Keep that allocation strictly local to a
            // node: cudaipc_create_tensor_list explicitly rejects a process
            // group larger than the visible device count and cannot open a
            // remote-node allocation.
            reduce_buffers_ = create_tensor_list(
                {max_ntokens_, n_dim_}, input_dtype_, context_, true);
            FTRAIN_CHECK(static_cast<int>(reduce_buffers_.size()) == world_size_);
            reduce_buffer_rows_ = max_ntokens_;

            if (a2a_signal_pipeline_enabled_)
            {
                // Each rank owns a counter and a ready epoch for every
                // logical N group.  Peers read the ready epochs through the
                // same IPC pointer list used by the A2A scratch buffers.
                a2a_signal_stride_ = max_num_groups_;
                a2a_signal_elems_ = 2 * a2a_signal_stride_;
                a2a_signal_buffers_ = create_tensor_list(
                    {a2a_signal_elems_},
                    ScalarType::Int,
                    context_,
                    true);
                FTRAIN_CHECK(static_cast<int>(a2a_signal_buffers_.size()) ==
                           world_size_);
                a2a_signal_buffers_[rank_].zero_();
                a2a_signal_host_ptrs_.resize(world_size_, nullptr);
                for (int peer = 0; peer < world_size_; ++peer)
                {
                    a2a_signal_host_ptrs_[peer] =
                        a2a_signal_buffers_[peer].data_ptr();
                }
            }

            std::vector<void *> reduce_ptrs(world_size_, nullptr);
            std::vector<void *> a2a_signal_ptrs;
            if (a2a_signal_pipeline_enabled_)
            {
                a2a_signal_ptrs.resize(world_size_, nullptr);
            }
            for (int peer = 0; peer < world_size_; ++peer)
            {
                reduce_ptrs[peer] = reduce_buffers_[peer].data_ptr();
                if (a2a_signal_pipeline_enabled_)
                {
                    a2a_signal_ptrs[peer] =
                        a2a_signal_buffers_[peer].data_ptr();
                }
            }
            CUDA_CHECK(hipMalloc(&d_reduce_buffer_ptrs_,
                                 reduce_ptrs.size() * sizeof(void *)));
            CUDA_CHECK(hipMemcpy(d_reduce_buffer_ptrs_,
                                 reduce_ptrs.data(),
                                 reduce_ptrs.size() * sizeof(void *),
                                 hipMemcpyHostToDevice));
            if (a2a_signal_pipeline_enabled_)
            {
                CUDA_CHECK(hipMalloc(&d_a2a_signal_buffer_ptrs_,
                                     a2a_signal_ptrs.size() *
                                         sizeof(void *)));
                CUDA_CHECK(hipMemcpy(d_a2a_signal_buffer_ptrs_,
                                     a2a_signal_ptrs.data(),
                                     a2a_signal_ptrs.size() * sizeof(void *),
                                     hipMemcpyHostToDevice));
            }
#ifndef FTRAIN_SHM_USE_NVSHMEM
            sync_buffers_ = create_tensor_list(
                {world_size_}, ScalarType::Int, context_, true);
            sync_buffers_[rank_].zero_();
            if (a2a_cooperative_fused_enabled_)
            {
                std::vector<int *> sync_ptrs(world_size_, nullptr);
                for (int peer = 0; peer < world_size_; ++peer)
                {
                    sync_ptrs[peer] = sync_buffers_[peer].data_ptr<int>();
                }
                CUDA_CHECK(hipMalloc(&d_sync_buffer_ptrs_,
                                     sync_ptrs.size() * sizeof(int *)));
                CUDA_CHECK(hipMemcpy(d_sync_buffer_ptrs_,
                                     sync_ptrs.data(),
                                     sync_ptrs.size() * sizeof(int *),
                                     hipMemcpyHostToDevice));
                // The cooperative group-pipeline uses a tail region of this
                // allocation as per-group ready epochs.  Keeping at least two
                // words at the front leaves state[0]/state[1] available for
                // the legacy non-cooperative barrier fallback.  Epoch values
                // are monotonic, so the ready words do not need a memset on
                // every launch.
                a2a_grid_barrier_ = zeros(
                    {2 * max_num_groups_}, int_options);
            }
#endif
        }
        else
        {
            // Cross-node reduction is performed by ProcessGroup/RCCL.  The
            // input to _reduce_scatter_base is local device memory containing
            // this rank's FP32 top-k stage cast back to the operator dtype.
            pg_reduce_buffer_ = empty(
                {max_ntokens_, n_dim_}, device_options.dtype(input_dtype_));
        }

        // NVIDIA/HCU keep the reduction path on a high-priority stream so a
        // ready N-chunk can make progress while the next AG GEMM tile is
        // running on the regular stream.  The priority is deliberately
        // configurable: setting it to 0 restores the old ordinary-stream
        // graph when comparing against a backup binary or when a particular
        // RCCL/DTK build has no useful priority scheduling.
        if (a2a_cu_partition_enabled_)
        {
            hipDeviceProp_t properties{};
            CUDA_CHECK(hipGetDeviceProperties(&properties, device));
            FTRAIN_CHECK(properties.multiProcessorCount >= 2);
            a2a_comm_cu_count_ =
                std::min(a2a_comm_cu_count_,
                         properties.multiProcessorCount - 1);
            const int mask_words =
                ceil_div(properties.multiProcessorCount, 32);
            std::vector<uint32_t> comm_mask(mask_words, 0u);
            std::vector<uint32_t> gemm_mask(mask_words, 0u);
            for (int cu = 0; cu < properties.multiProcessorCount; ++cu)
            {
                std::vector<uint32_t> &mask =
                    cu < a2a_comm_cu_count_ ? comm_mask : gemm_mask;
                mask[cu / 32] |= uint32_t{1} << (cu % 32);
            }
            CUDA_CHECK(hipExtStreamCreateWithCUMask(
                &gemm_stream_, mask_words, gemm_mask.data()));
            CUDA_CHECK(hipExtStreamCreateWithCUMask(
                &reduce_stream_, mask_words, comm_mask.data()));
            if (a2a_signal_pipeline_enabled_ ||
                a2a_admitted_scatter_enabled_ ||
                a2a_adaptive_admission_enabled_ ||
                a2a_decoupled_scatter_enabled_)
            {
                CUDA_CHECK(hipExtStreamCreateWithCUMask(
                    &a2a_scatter_stream_, mask_words, comm_mask.data()));
            }
        }
        else
        {
            CUDA_CHECK(hipStreamCreateWithFlags(&gemm_stream_,
                                                hipStreamNonBlocking));
            if (get_bool_from_env(
                    "FTRAIN_MOE_RS_REDUCE_STREAM_HIGH_PRIORITY", true))
            {
                int least_priority = 0;
                int greatest_priority = 0;
                CUDA_CHECK(hipDeviceGetStreamPriorityRange(&least_priority,
                                                           &greatest_priority));
                CUDA_CHECK(hipStreamCreateWithPriority(&reduce_stream_,
                                                       hipStreamNonBlocking,
                                                       greatest_priority));
            }
            else
            {
                CUDA_CHECK(hipStreamCreateWithFlags(&reduce_stream_,
                                                    hipStreamNonBlocking));
            }
            if (a2a_signal_pipeline_enabled_ ||
                a2a_admitted_scatter_enabled_ ||
                a2a_adaptive_admission_enabled_ ||
                a2a_decoupled_scatter_enabled_)
            {
                // Scatter is intentionally left on an ordinary-priority
                // stream; the reduce stream remains the high-priority sum
                // consumer.
                CUDA_CHECK(hipStreamCreateWithFlags(&a2a_scatter_stream_,
                                                    hipStreamNonBlocking));
            }
        }
        CUDA_CHECK(
            hipEventCreateWithFlags(&input_ready_event_, hipEventDisableTiming));
        CUDA_CHECK(
            hipEventCreateWithFlags(&flags_reset_event_, hipEventDisableTiming));
        CUDA_CHECK(
            hipEventCreateWithFlags(&gemm_done_event_, hipEventDisableTiming));
        CUDA_CHECK(
            hipEventCreateWithFlags(&reduce_done_event_, hipEventDisableTiming));
        if (a2a_signal_pipeline_enabled_ ||
            a2a_admitted_scatter_enabled_ ||
            a2a_adaptive_admission_enabled_ ||
            a2a_decoupled_scatter_enabled_)
        {
            CUDA_CHECK(hipEventCreateWithFlags(&a2a_scatter_done_event_,
                                               hipEventDisableTiming));
        }
        host_expert_counts_.resize(num_experts_);
        host_expert_offsets_.resize(num_experts_ + 1);
        host_expert_rank_bounds_.resize(num_experts_ * 2);
    }

    ~MoEReduceRSImpl()
    {
        if (gemm_stream_ != nullptr)
        {
            hipStreamSynchronize(gemm_stream_);
        }
        if (reduce_stream_ != nullptr)
        {
            hipStreamSynchronize(reduce_stream_);
        }
        if (a2a_scatter_stream_ != nullptr)
        {
            hipStreamSynchronize(a2a_scatter_stream_);
        }
        if (input_ready_event_ != nullptr)
        {
            hipEventDestroy(input_ready_event_);
        }
        if (flags_reset_event_ != nullptr)
        {
            hipEventDestroy(flags_reset_event_);
        }
        if (gemm_done_event_ != nullptr)
        {
            hipEventDestroy(gemm_done_event_);
        }
        if (reduce_done_event_ != nullptr)
        {
            hipEventDestroy(reduce_done_event_);
        }
        if (a2a_scatter_done_event_ != nullptr)
        {
            hipEventDestroy(a2a_scatter_done_event_);
        }
        if (gemm_stream_ != nullptr)
        {
            hipStreamDestroy(gemm_stream_);
        }
        if (reduce_stream_ != nullptr)
        {
            hipStreamDestroy(reduce_stream_);
        }
        if (a2a_scatter_stream_ != nullptr)
        {
            hipStreamDestroy(a2a_scatter_stream_);
        }
        if (d_reduce_buffer_ptrs_ != nullptr)
        {
            hipFree(d_reduce_buffer_ptrs_);
        }
        if (d_a2a_signal_buffer_ptrs_ != nullptr)
        {
            hipFree(d_a2a_signal_buffer_ptrs_);
        }
        if (d_rank_m_offsets_ != nullptr)
        {
            hipFree(d_rank_m_offsets_);
        }
        if (d_sync_buffer_ptrs_ != nullptr)
        {
            hipFree(d_sync_buffer_ptrs_);
        }
    }

    Tensor run_a2a(Tensor input,
                          Tensor weight,
                          Tensor chosen_experts,
                          Tensor expert_weight,
                          std::optional<Tensor> output)
    {
        return run_impl(std::move(input),
                        std::move(weight),
                        std::move(chosen_experts),
                        std::move(expert_weight),
                        std::move(output),
                        false);
    }

    Tensor run_a2a_async(Tensor input,
                                Tensor weight,
                                Tensor chosen_experts,
                                Tensor expert_weight,
                                std::optional<Tensor> output)
    {
        return run_impl(std::move(input),
                        std::move(weight),
                        std::move(chosen_experts),
                        std::move(expert_weight),
                        std::move(output),
                        true);
    }

    void wait_a2a()
    {
        if (!a2a_async_pending_)
        {
            return;
        }
        CUDA_CHECK(hipEventSynchronize(reduce_done_event_));
        hipStream_t current_stream =
            context_->stream();
        CUDA_CHECK(
            hipStreamWaitEvent(current_stream, reduce_done_event_, 0));
        pending_a2a_input_ = Tensor();
        pending_a2a_weight_ = Tensor();
        pending_a2a_chosen_experts_ = Tensor();
        pending_a2a_expert_weight_ = Tensor();
        pending_a2a_output_ = Tensor();
        a2a_async_pending_ = false;
    }

    Tensor n_chunk_done_flags()
    {
        return n_chunk_done_flags_;
    }

    int32_t last_chunk_size_n() const
    {
        return last_chunk_size_n_;
    }

    int32_t last_num_groups() const
    {
        return last_num_groups_;
    }

    bool last_n_chunk_signal() const
    {
        return last_n_chunk_signal_;
    }

    void set_rank_m_sizes(int32_t active_ntokens,
                          std::vector<int64_t> rank_m_sizes)
    {
        FTRAIN_CHECK(active_ntokens > 0 && active_ntokens <= max_ntokens_)
            << "active_ntokens exceeds MoEReduceRS capacity";
        FTRAIN_CHECK(static_cast<int>(rank_m_sizes.size()) == world_size_)
            << "rank_m_sizes must contain one entry per rank";
        int64_t total = 0;
        for (int64_t rows : rank_m_sizes)
        {
            FTRAIN_CHECK(rows >= 0) << "rank_m_sizes must be non-negative";
            total += rows;
        }
        FTRAIN_CHECK(total == active_ntokens)
            << "sum(rank_m_sizes) must equal active_ntokens";
        std::vector<int> rank_m_offsets(world_size_ + 1, 0);
        int64_t max_rows = 0;
        for (int peer = 0; peer < world_size_; ++peer)
        {
            FTRAIN_CHECK(rank_m_sizes[peer] <= std::numeric_limits<int>::max())
                << "rank M exceeds the native A2A index range";
            rank_m_offsets[peer + 1] =
                rank_m_offsets[peer] + static_cast<int>(rank_m_sizes[peer]);
            max_rows = std::max(max_rows, rank_m_sizes[peer]);
        }
        FTRAIN_CHECK(rank_m_offsets.back() == active_ntokens)
            << "rank M offsets overflowed the active token count";

        // The native ragged A2A layout reserves one compact local-M segment
        // per producer rank in every destination IPC buffer.  Most cached
        // training operators already have a power-of-two max_ntokens capacity
        // large enough for this; grow only when a small standalone operator
        // is constructed at exactly the active ragged M.
        ensure_ragged_reduce_buffer_rows(
            static_cast<int64_t>(world_size_) * max_rows);
        CUDA_CHECK(hipMemcpy(d_rank_m_offsets_,
                             rank_m_offsets.data(),
                             rank_m_offsets.size() * sizeof(int),
                             hipMemcpyHostToDevice));
        active_ntokens_ = active_ntokens;
        rank_m_sizes_ = std::move(rank_m_sizes);
        rank_m_offsets_ = std::move(rank_m_offsets);
    }

private:
    void ensure_ragged_reduce_buffer_rows(int64_t required_rows)
    {
        if (nnodes_ != 1 || required_rows <= reduce_buffer_rows_)
        {
            return;
        }
        FTRAIN_CHECK(required_rows > 0)
            << "ragged A2A reduce-buffer capacity must be positive";
        FTRAIN_CHECK(d_reduce_buffer_ptrs_ != nullptr)
            << "ragged A2A IPC pointer table was not initialized";

        // This runs only when a cached operator first observes a larger
        // ragged destination segment.  Do not release a peer-visible mapping
        // before all streams on this rank have finished the prior invocation.
        CUDA_CHECK(hipStreamSynchronize(gemm_stream_));
        CUDA_CHECK(hipStreamSynchronize(reduce_stream_));
        if (a2a_scatter_stream_ != nullptr)
        {
            CUDA_CHECK(hipStreamSynchronize(a2a_scatter_stream_));
        }

        // create_tensor_list performs a ProcessGroup handle exchange, so
        // every rank receives a fresh symmetric IPC mapping before the device
        // pointer table below is published to a kernel.
        std::vector<Tensor> old_buffers = std::move(reduce_buffers_);
        reduce_buffers_ = create_tensor_list(
            {required_rows, n_dim_}, input_dtype_, context_, true);
        FTRAIN_CHECK(static_cast<int>(reduce_buffers_.size()) == world_size_);
        std::vector<void *> reduce_ptrs(world_size_, nullptr);
        for (int peer = 0; peer < world_size_; ++peer)
        {
            reduce_ptrs[peer] = reduce_buffers_[peer].data_ptr();
        }
        CUDA_CHECK(hipMemcpy(d_reduce_buffer_ptrs_,
                             reduce_ptrs.data(),
                             reduce_ptrs.size() * sizeof(void *),
                             hipMemcpyHostToDevice));
        reduce_buffer_rows_ = required_rows;
    }

    void validate_inputs(const Tensor &input,
                         const Tensor &weight,
                         const Tensor &chosen_experts,
                         const Tensor &expert_weight) const
    {
        CHECK_INPUT(input, input_dtype_);
        CHECK_INPUT(weight, input_dtype_);
        CHECK_INPUT(expert_weight, input_dtype_);
        CHECK_INPUT_LOOSE(chosen_experts);
        FTRAIN_CHECK(chosen_experts.scalar_type() == ScalarType::Int ||
                   chosen_experts.scalar_type() == ScalarType::Long)
            << "chosen_experts must be int32 or int64";
        FTRAIN_CHECK(chosen_experts.dim() == 2);
        const int ntokens = chosen_experts.size(0);
        FTRAIN_CHECK(ntokens > 0 && ntokens <= max_ntokens_);
        if (rank_m_sizes_.empty())
        {
            FTRAIN_CHECK(ntokens % world_size_ == 0)
                << "equal-split MoEReduceRS requires M divisible by world";
        }
        else
        {
            FTRAIN_CHECK(active_ntokens_ == ntokens)
                << "set_rank_m_sizes must be called for the active M";
        }
        CHECK_2D(chosen_experts, ntokens, topk_);
        CHECK_2D(expert_weight, ntokens, topk_);
        CHECK_2D(input, ntokens * topk_, k_dim_);
        if (active_weight_native_kn_)
        {
            FTRAIN_CHECK(weight.dim() == 3 && weight.size(0) == num_experts_ &&
                       weight.size(1) == k_dim_ && weight.size(2) == n_dim_)
                << "native-KN weight must be [E, K_local, N], got "
                << weight.sizes();
        }
        else
        {
            FTRAIN_CHECK(weight.dim() == 3 && weight.size(0) == num_experts_ &&
                       weight.size(1) == n_dim_ && weight.size(2) == k_dim_)
                << "weight must be [E, N, K_local], got " << weight.sizes();
        }
    }

    Tensor make_output(int ntokens,
                              std::optional<Tensor> output) const
    {
        const int local_rows = local_output_rows(ntokens);
        if (output.has_value())
        {
            Tensor out = output.value();
            CHECK_INPUT(out, input_dtype_);
            FTRAIN_CHECK(out.dim() == 2 && out.size(0) >= local_rows &&
                       out.size(1) == n_dim_);
            return out.narrow(0, 0, local_rows);
        }
        if (ntokens == max_ntokens_ &&
            local_rows <= output_buffer_.size(0))
        {
            return output_buffer_.narrow(0, 0, local_rows);
        }
        return empty(
            {local_rows, n_dim_},
            context_->tensor_options()
                .dtype(input_dtype_)
                .device_index(context_->device()));
    }

    int local_output_rows(int ntokens) const
    {
        if (rank_m_sizes_.empty())
        {
            FTRAIN_CHECK(ntokens % world_size_ == 0);
            return ntokens / world_size_;
        }
        FTRAIN_CHECK(active_ntokens_ == ntokens);
        return static_cast<int>(rank_m_sizes_[rank_]);
    }

    bool has_ragged_partition(int ntokens) const
    {
        if (rank_m_sizes_.empty())
        {
            return false;
        }
        if (ntokens % world_size_ != 0)
        {
            return true;
        }
        const int equal_rows = ntokens / world_size_;
        return std::any_of(rank_m_sizes_.begin(),
                           rank_m_sizes_.end(),
                           [equal_rows](int64_t rows) {
                               return rows != equal_rows;
        });
    }

    int prepare_metadata(const Tensor &chosen_experts,
                         int ntokens,
                         hipStream_t stream)
    {
        const bool cache_enabled = get_bool_from_env(
            "FTRAIN_MOE_RS_METADATA_CACHE", true);
        const int64_t metadata_version = chosen_experts.version();
        const bool cache_hit =
            cache_enabled && metadata_cache_valid_ &&
            metadata_topk_tensor_.is_same(chosen_experts) &&
            metadata_topk_src_ == chosen_experts.data_ptr() &&
            metadata_topk_version_ == metadata_version &&
            metadata_topk_ntokens_ == ntokens &&
            metadata_topk_dtype_ == chosen_experts.scalar_type();

        if (cache_hit)
        {
            return metadata_rank_tile_count_;
        }

        ag_group_gemm_prepare_metadata_flat_input_impl(
            chosen_experts.data_ptr(),
            source_sorted_token_ids_.data_ptr<int>(),
            source_sorted_gather_index_.data_ptr<int>(),
            source_expert_rank_counts_.data_ptr<int>(),
            source_expert_rank_counters_.data_ptr<int>(),
            source_expert_rank_offsets_.data_ptr<int>(),
            source_global_expert_offsets_.data_ptr<int>(),
            source_rank_cursors_.data_ptr<int>(),
            source_rank_tile_info_.data_ptr<int>(),
            source_tile_experts_.data_ptr<int>(),
            source_tile_starts_.data_ptr<int>(),
            source_tile_rank_starts_.data_ptr<int>(),
            source_tile_rank_ends_.data_ptr<int>(),
            nullptr,
            source_ntiles_.data_ptr<int>(),
            ntokens,
            topk_,
            num_experts_,
            1,
            0,
            source_sorted_capacity_,
            max_rank_tiles_,
            0,
            0,
            false,
            false,
            chosen_experts.scalar_type(),
            stream);

        int rank_tile_count = 0;
        CUDA_CHECK(hipMemcpyAsync(&rank_tile_count,
                                  source_ntiles_.data_ptr<int>(),
                                  sizeof(int),
                                  hipMemcpyDeviceToHost,
                                  stream));
        CUDA_CHECK(hipStreamSynchronize(stream));
        FTRAIN_CHECK(rank_tile_count > 0 && rank_tile_count <= max_rank_tiles_)
            << "invalid source MT256 rank tile count " << rank_tile_count;
        metadata_topk_tensor_ = chosen_experts;
        metadata_cache_valid_ = true;
        metadata_topk_src_ = chosen_experts.data_ptr();
        metadata_topk_version_ = metadata_version;
        metadata_topk_ntokens_ = ntokens;
        metadata_topk_dtype_ = chosen_experts.scalar_type();
        metadata_rank_tile_count_ = rank_tile_count;
        return rank_tile_count;
    }

    void launch_grouped_gemm(const Tensor &input,
                             const Tensor &weight,
                             const Tensor &chosen_experts,
                             const Tensor &expert_weight,
                             int ntokens,
                             int rank_tile_count,
                             const ChunkConfig &chunks,
                             bool n_chunk_signal,
                             bool topk_stage)
    {
        (void)chosen_experts;
        if (n_chunk_signal)
        {
            // The recovered AG rank-tile object exposes completion flags per
            // logical N chunk, while its tile-N is fixed at 256 columns.  Its
            // ABI validates exactly this relationship before launching the
            // kernel; keep the same check here so a future chunk policy
            // cannot silently wait on the wrong flag group.
            FTRAIN_CHECK(chunks.chunk_size_n > 0 &&
                       chunks.chunk_size_n % 256 == 0)
                << "AG N-chunk signal requires chunk_size_n to be a 256 "
                   "column multiple, got "
                << chunks.chunk_size_n;
            FTRAIN_CHECK(chunks.num_groups ==
                       ceil_div(n_dim_, chunks.chunk_size_n))
                << "AG N-chunk signal group mapping mismatch: n_dim=" << n_dim_
                << " chunk_size_n=" << chunks.chunk_size_n
                << " groups=" << chunks.num_groups;
        }
        if (!fc2_packed_input_.defined())
        {
            fc2_packed_input_ =
                empty({source_sorted_capacity_, k_dim_}, input.options());
            fc2_contract_info_ =
                empty({max_rank_tiles_ * 5}, source_rank_tile_info_.options());
        }
        moe_fc2_prepare_input(input.data_ptr(),
                              fc2_packed_input_.data_ptr(),
                              source_sorted_gather_index_.data_ptr<int>(),
                              source_rank_tile_info_.data_ptr<int>(),
                              fc2_contract_info_.data_ptr<int>(),
                              source_sorted_capacity_,
                              static_cast<int>(input.size(0)),
                              k_dim_,
                              rank_tile_count,
                              gemm_stream_);
        moe_fc2_gemm_forward(
            fc2_packed_input_.data_ptr(),
            weight.data_ptr(),
            source_sorted_token_ids_.data_ptr<int>(),
            fc2_contract_info_.data_ptr<int>(),
            gemm_output_.data_ptr(),
            n_chunk_signal
                ? (topk_stage ? gemm_done_counts_.data_ptr<int>()
                              : n_chunk_done_counts_.data_ptr<int>())
                : nullptr,
            n_chunk_signal
                ? (topk_stage ? gemm_done_flags_.data_ptr<int>()
                              : n_chunk_done_flags_.data_ptr<int>())
                : nullptr,
            rank_tile_count,
            ntokens * topk_,
            n_dim_,
            k_dim_,
            chunks.chunk_size_n,
            n_chunk_signal ? chunks.num_groups : 0,
            gemm_stream_,
            !active_weight_native_kn_);

        if (topk_stage)
        {
            moe_reduce_rs_build_topk_stage(
                gemm_output_.data_ptr(),
                expert_weight.data_ptr(),
                topk_stage_.data_ptr(),
                n_chunk_signal ? gemm_done_flags_.data_ptr<int>() : nullptr,
                n_chunk_signal ? n_chunk_done_counts_.data_ptr<int>() : nullptr,
                n_chunk_signal ? n_chunk_done_flags_.data_ptr<int>() : nullptr,
                ntokens,
                n_dim_,
                topk_,
                chunks.chunk_size_n,
                chunks.num_groups,
                1,
                input_dtype_,
                gemm_stream_);
        }
    }

    void sync_reduce_stream()
    {
#ifdef FTRAIN_SHM_USE_NVSHMEM
        all_ranks_sync(reduce_stream_, std::nullopt, std::nullopt);
#else
        all_ranks_sync(reduce_stream_, sync_buffers_, rank_);
#endif
    }


    int next_a2a_epoch()
    {
        FTRAIN_CHECK(a2a_signal_pipeline_enabled_ &&
                   !a2a_signal_buffers_.empty() &&
                   a2a_signal_elems_ > 0);
        if (a2a_epoch_ >= 0x7ffffffe)
        {
            // The final sum/lifetime barrier from the previous invocation is
            // already ordered before the next host call.  Clear both counter
            // and ready regions only at this practically unreachable epoch
            // wrap boundary, then restart at epoch 1.
            CUDA_CHECK(hipStreamSynchronize(reduce_stream_));
            CUDA_CHECK(hipStreamSynchronize(a2a_scatter_stream_));
            CUDA_CHECK(hipMemset(a2a_signal_buffers_[rank_].data_ptr<int>(),
                                 0,
                                 static_cast<size_t>(a2a_signal_elems_) *
                                     sizeof(int)));
            a2a_epoch_ = 0;
        }
        return ++a2a_epoch_;
    }

    int next_a2a_cooperative_epoch()
    {
        // Group-pipeline ready words are tagged with a monotonically
        // increasing invocation epoch, so no steady-state memset is needed.
        // Clear only at the practically unreachable signed-int wrap boundary.
        if (a2a_cooperative_epoch_ >= 0x7ffffffe)
        {
            CUDA_CHECK(hipStreamSynchronize(reduce_stream_));
            if (a2a_grid_barrier_.defined())
            {
                a2a_grid_barrier_.zero_();
            }
            a2a_cooperative_epoch_ = 0;
        }
        return ++a2a_cooperative_epoch_;
    }



    void launch_reduce_a2a_ragged_signal(Tensor &output,
                                         const Tensor &expert_weight,
                                         int ntokens,
                                         const ChunkConfig &chunks,
                                         bool topk_stage,
                                         bool scale_in_reduce)
    {
        FTRAIN_CHECK(a2a_scatter_stream_ != nullptr &&
                   a2a_scatter_done_event_ != nullptr &&
                   d_a2a_signal_buffer_ptrs_ != nullptr &&
                   d_rank_m_offsets_ != nullptr)
            << "ragged A2A signal resources were not initialized";
        FTRAIN_CHECK(!rank_m_sizes_.empty() &&
                   static_cast<int>(rank_m_offsets_.size()) == world_size_ + 1)
            << "ragged A2A requires rank M offsets";
        FTRAIN_CHECK(chunks.n_chunk_signal)
            << "ragged A2A requires per-N-chunk AG completion signals";

        const int epoch = next_a2a_epoch();
        CUDA_CHECK(hipStreamWaitEvent(a2a_scatter_stream_,
                                      flags_reset_event_,
                                      0));
        CUDA_CHECK(hipMemsetAsync(
            a2a_signal_buffers_[rank_].data_ptr<int>(),
            0,
            static_cast<size_t>(a2a_signal_stride_) * sizeof(int),
            a2a_scatter_stream_));

        const void *reduce_scale =
            !topk_stage && scale_in_reduce ? expert_weight.data_ptr() : nullptr;
        // Same policy split as the fixed-M route: the shape policy decides
        // whether the hardware-wait scheduling wins, and everything else runs
        // the persistent-producer stream-wait variant instead of failing.
        auto launch_ragged_pipeline = [&](auto pipeline_fn) {
            pipeline_fn(topk_stage ? nullptr : gemm_output_.data_ptr(),
                        reduce_scale,
                        topk_stage ? topk_stage_.data_ptr() : nullptr,
                        d_reduce_buffer_ptrs_,
                        d_a2a_signal_buffer_ptrs_,
                        a2a_signal_host_ptrs_.data(),
                        n_chunk_done_flags_.data_ptr<int>(),
                        d_rank_m_offsets_,
                        output.data_ptr(),
                        ntokens,
                        local_output_rows(ntokens),
                        n_dim_,
                        rank_,
                        world_size_,
                        topk_,
                        chunks.chunk_size_n,
                        chunks.num_groups,
                        a2a_signal_stride_,
                        epoch,
                        a2a_signal_producer_block_budget_,
                        a2a_signal_consumer_block_budget_,
                        input_dtype_,
                        a2a_scatter_stream_,
                        reduce_stream_);
        };
        if (a2a_hardware_wait_enabled_)
        {
            launch_ragged_pipeline(
                moe_reduce_rs_a2a_signal_pipeline_ragged_hardware_wait);
        }
        else
        {
            launch_ragged_pipeline(
                moe_reduce_rs_a2a_signal_pipeline_ragged_stream_wait);
        }
        CUDA_CHECK(hipEventRecord(a2a_scatter_done_event_,
                                  a2a_scatter_stream_));
        CUDA_CHECK(hipStreamWaitEvent(reduce_stream_,
                                      a2a_scatter_done_event_,
                                      0));
        sync_reduce_stream();
    }

    // The fixed-M cooperative grid is materially faster than the separate
    // ragged producer/consumer pipeline.  Keep this route opt-in while it is
    // being validated: it shares the same IPC barriers as fixed-M, but maps
    // gathered tokens through rank_m_offsets and writes the compact
    // [producer_rank][destination_local_row] ragged scratch layout.
    void launch_reduce_a2a_ragged_cooperative(
        Tensor &output,
        const Tensor &expert_weight,
        int ntokens,
        const ChunkConfig &chunks,
        bool topk_stage,
        bool scale_in_reduce)
    {
        FTRAIN_CHECK(d_sync_buffer_ptrs_ != nullptr &&
                   a2a_grid_barrier_.defined() &&
                   d_rank_m_offsets_ != nullptr)
            << "ragged cooperative A2A resources were not initialized";
        FTRAIN_CHECK(!rank_m_sizes_.empty() &&
                   static_cast<int>(rank_m_offsets_.size()) == world_size_ + 1)
            << "ragged cooperative A2A requires rank M offsets";
        const void *reduce_scale =
            !topk_stage && scale_in_reduce ? expert_weight.data_ptr() : nullptr;
        // The cooperative group-pipeline reduction still assumes a uniform
        // source-row stride.  The non-group pipeline below preserves the
        // fixed-M fused scatter/IPC contract while using ragged strides.
        moe_reduce_rs_a2a_cooperative_fused(
            topk_stage ? nullptr : gemm_output_.data_ptr(),
            reduce_scale,
            topk_stage ? topk_stage_.data_ptr() : nullptr,
            d_reduce_buffer_ptrs_,
            d_sync_buffer_ptrs_,
            a2a_grid_barrier_.data_ptr<int>(),
            n_chunk_done_flags_.data_ptr<int>(),
            output.data_ptr(),
            d_rank_m_offsets_,
            local_output_rows(ntokens),
            ntokens,
            n_dim_,
            rank_,
            world_size_,
            topk_,
            chunks.chunk_size_n,
            chunks.num_groups,
            next_a2a_cooperative_epoch(),
            input_dtype_,
            a2a_hip_cooperative_launch_enabled_,
            false,
            reduce_stream_);
    }

    void launch_reduce_a2a(Tensor &output,
                           const Tensor &expert_weight,
                           int ntokens,
                           int rank_tile_count,
                           const ChunkConfig &chunks,
                           bool topk_stage,
                           bool scale_in_reduce)
    {
        const bool direct_scatter_override =
            !topk_stage &&
            (a2a_paced_scatter_enabled_ || a2a_admitted_scatter_enabled_ ||
             a2a_adaptive_admission_enabled_);
        if (!direct_scatter_override &&
            (a2a_signal_pipeline_enabled_ || a2a_hardware_wait_enabled_) &&
            chunks.n_chunk_signal)
        {
            FTRAIN_CHECK(a2a_scatter_stream_ != nullptr &&
                       a2a_scatter_done_event_ != nullptr &&
                       d_a2a_signal_buffer_ptrs_ != nullptr);
            const int epoch = next_a2a_epoch();
            // Reset only this rank's CTA counters.  Ready epochs are not
            // cleared; the monotonically increasing epoch prevents a sum
            // kernel from consuming the previous invocation's signal.
            CUDA_CHECK(hipStreamWaitEvent(a2a_scatter_stream_,
                                          flags_reset_event_,
                                          0));
            CUDA_CHECK(hipMemsetAsync(
                a2a_signal_buffers_[rank_].data_ptr<int>(),
                0,
                static_cast<size_t>(a2a_signal_stride_) * sizeof(int),
                a2a_scatter_stream_));

            const void *reduce_scale =
                !topk_stage && scale_in_reduce
                    ? expert_weight.data_ptr()
                    : nullptr;
            if (a2a_split_pipeline_enabled_)
            {
                moe_reduce_rs_a2a_signal_pipeline_split(
                    topk_stage ? nullptr : gemm_output_.data_ptr(),
                    reduce_scale,
                    topk_stage ? topk_stage_.data_ptr() : nullptr,
                    d_reduce_buffer_ptrs_,
                    d_a2a_signal_buffer_ptrs_,
                    n_chunk_done_flags_.data_ptr<int>(),
                    output.data_ptr(),
                    ntokens,
                    n_dim_,
                    rank_,
                    world_size_,
                    topk_,
                    chunks.chunk_size_n,
                    chunks.num_groups,
                    a2a_signal_stride_,
                    epoch,
                    a2a_signal_producer_block_budget_,
                    a2a_signal_consumer_block_budget_,
                    input_dtype_,
                    a2a_scatter_stream_,
                    reduce_stream_);
            }
            else if (a2a_hardware_wait_enabled_)
            {
                const bool use_staged_block_budget =
                    a2a_staged_block_budget_enabled_ &&
                    !active_weight_native_kn_;
                moe_reduce_rs_a2a_signal_pipeline_hardware_wait(
                    topk_stage ? nullptr : gemm_output_.data_ptr(),
                    reduce_scale,
                    topk_stage ? topk_stage_.data_ptr() : nullptr,
                    d_reduce_buffer_ptrs_,
                    d_a2a_signal_buffer_ptrs_,
                    a2a_signal_host_ptrs_.data(),
                    n_chunk_done_flags_.data_ptr<int>(),
                    output.data_ptr(),
                    ntokens,
                    n_dim_,
                    rank_,
                    world_size_,
                    topk_,
                    chunks.chunk_size_n,
                    chunks.num_groups,
                    a2a_signal_stride_,
                    epoch,
                    a2a_signal_producer_block_budget_,
                    a2a_signal_consumer_block_budget_,
                    use_staged_block_budget
                        ? a2a_signal_early_producer_block_budget_
                        : a2a_signal_producer_block_budget_,
                    use_staged_block_budget
                        ? a2a_signal_early_consumer_block_budget_
                        : a2a_signal_consumer_block_budget_,
                    use_staged_block_budget
                        ? a2a_signal_full_budget_start_group_
                        : 0,
                    a2a_wave64_512_scatter_enabled_,
                    a2a_wave64_512_sum_enabled_,
                    input_dtype_,
                    a2a_scatter_stream_,
                    reduce_stream_);
            }
            else if (a2a_signal_stream_wait_enabled_)
            {
                moe_reduce_rs_a2a_signal_pipeline_stream_wait(
                    topk_stage ? nullptr : gemm_output_.data_ptr(),
                    reduce_scale,
                    topk_stage ? topk_stage_.data_ptr() : nullptr,
                    d_reduce_buffer_ptrs_,
                    d_a2a_signal_buffer_ptrs_,
                    a2a_signal_host_ptrs_.data(),
                    n_chunk_done_flags_.data_ptr<int>(),
                    output.data_ptr(),
                    ntokens,
                    n_dim_,
                    rank_,
                    world_size_,
                    topk_,
                    chunks.chunk_size_n,
                    chunks.num_groups,
                    a2a_signal_stride_,
                    epoch,
                    a2a_signal_block_budget_,
                    input_dtype_,
                    a2a_scatter_stream_,
                    reduce_stream_);
            }
            else
            {
                moe_reduce_rs_a2a_signal_pipeline(
                    topk_stage ? nullptr : gemm_output_.data_ptr(),
                    reduce_scale,
                    topk_stage ? topk_stage_.data_ptr() : nullptr,
                    d_reduce_buffer_ptrs_,
                    d_a2a_signal_buffer_ptrs_,
                    n_chunk_done_flags_.data_ptr<int>(),
                    output.data_ptr(),
                    ntokens,
                    n_dim_,
                    rank_,
                    world_size_,
                    topk_,
                    chunks.chunk_size_n,
                    chunks.num_groups,
                    a2a_signal_stride_,
                    epoch,
                    a2a_signal_block_budget_,
                    input_dtype_,
                    a2a_scatter_stream_,
                    reduce_stream_);
            }

            // The sum kernel may finish its last group as soon as the last
            // ready epoch is published, while the scatter kernel is still
            // unwinding.  Order the final lifetime barrier after both kernels
            // without inserting a wait before the sum launch.
            CUDA_CHECK(hipEventRecord(a2a_scatter_done_event_,
                                      a2a_scatter_stream_));
            CUDA_CHECK(hipStreamWaitEvent(reduce_stream_,
                                          a2a_scatter_done_event_,
                                          0));
            sync_reduce_stream();
            return;
        }

        if (!direct_scatter_override && a2a_cooperative_fused_enabled_)
        {
            FTRAIN_CHECK(d_sync_buffer_ptrs_ != nullptr &&
                       a2a_grid_barrier_.defined());
            const void *reduce_scale =
                !topk_stage && scale_in_reduce
                    ? expert_weight.data_ptr()
                    : nullptr;
            const bool group_pipeline =
                a2a_group_pipeline_enabled_ && chunks.n_chunk_signal &&
                a2a_hip_cooperative_launch_enabled_;
            moe_reduce_rs_a2a_cooperative_fused(
                topk_stage ? nullptr : gemm_output_.data_ptr(),
                reduce_scale,
                topk_stage ? topk_stage_.data_ptr() : nullptr,
                d_reduce_buffer_ptrs_,
                d_sync_buffer_ptrs_,
                a2a_grid_barrier_.data_ptr<int>(),
                n_chunk_done_flags_.data_ptr<int>(),
                output.data_ptr(),
                nullptr,
                0,
                ntokens,
                n_dim_,
                rank_,
                world_size_,
                topk_,
                chunks.chunk_size_n,
                chunks.num_groups,
                next_a2a_cooperative_epoch(),
                input_dtype_,
                a2a_hip_cooperative_launch_enabled_,
                group_pipeline,
                reduce_stream_);
            return;
        }

        if (topk_stage)
        {
            moe_reduce_rs_topk_stage_scatter_a2a(
                topk_stage_.data_ptr(),
                d_reduce_buffer_ptrs_,
                n_chunk_done_flags_.data_ptr<int>(),
                ntokens,
                n_dim_,
                rank_,
                world_size_,
                chunks.chunk_size_n,
                chunks.num_groups,
                input_dtype_,
                a2a_decoupled_scatter_enabled_ ? a2a_scatter_stream_
                                               : reduce_stream_);
            if (a2a_decoupled_scatter_enabled_)
            {
                CUDA_CHECK(hipEventRecord(a2a_scatter_done_event_,
                                          a2a_scatter_stream_));
                CUDA_CHECK(hipStreamWaitEvent(reduce_stream_,
                                              a2a_scatter_done_event_,
                                              0));
            }
            sync_reduce_stream();
            moe_reduce_rs_sum_a2a(d_reduce_buffer_ptrs_,
                                  output.data_ptr(),
                                  ntokens / world_size_,
                                  n_dim_,
                                  rank_,
                                  world_size_,
                                  input_dtype_,
                                  reduce_stream_);
            sync_reduce_stream();
            return;
        }

        const void *reduce_scale =
            scale_in_reduce ? expert_weight.data_ptr() : nullptr;
        if (a2a_decoupled_scatter_enabled_)
        {
            FTRAIN_CHECK(a2a_scatter_stream_ != nullptr &&
                       a2a_scatter_done_event_ != nullptr);
            CUDA_CHECK(hipStreamWaitEvent(a2a_scatter_stream_,
                                          flags_reset_event_,
                                          0));
        }
        const bool per_chunk =
            chunks.num_groups >= 2 &&
            get_bool_from_env("FTRAIN_MOE_RS_A2A_PER_CHUNK", false);
        if (per_chunk)
        {
            for (int group = 0; group < chunks.num_groups; ++group)
            {
                const int chunk_offset_n = group * chunks.chunk_size_n;
                const int chunk_size_n =
                    std::min(chunks.chunk_size_n, n_dim_ - chunk_offset_n);
                moe_reduce_rs_topk_reduce_a2a_chunk(
                    gemm_output_.data_ptr(),
                    reduce_scale,
                    d_reduce_buffer_ptrs_,
                    n_chunk_done_flags_.data_ptr<int>(),
                    ntokens,
                    n_dim_,
                    rank_,
                    world_size_,
                    topk_,
                    group,
                    chunk_offset_n,
                    chunk_size_n,
                    input_dtype_,
                    reduce_stream_);
                sync_reduce_stream();
                moe_reduce_rs_sum_a2a_chunk(d_reduce_buffer_ptrs_,
                                            output.data_ptr(),
                                            ntokens / world_size_,
                                            n_dim_,
                                            rank_,
                                            world_size_,
                                            chunk_offset_n,
                                            chunk_size_n,
                                            input_dtype_,
                                            reduce_stream_);
                sync_reduce_stream();
            }
            return;
        }

        const bool adaptive_admission =
            a2a_adaptive_admission_enabled_ &&
            !a2a_decoupled_scatter_enabled_ && chunks.n_chunk_signal &&
            chunks.num_groups >= 2 && v36_task_control_.defined();
        const bool admitted_scatter =
            a2a_admitted_scatter_enabled_ &&
            !a2a_decoupled_scatter_enabled_ && !adaptive_admission &&
            chunks.num_groups >= 2;
        const bool paced_scatter =
            a2a_paced_scatter_enabled_ && !a2a_decoupled_scatter_enabled_ &&
            !adaptive_admission && !admitted_scatter &&
            chunks.num_groups >= 2;
        const bool scheduled_scatter =
            !a2a_decoupled_scatter_enabled_ && chunks.num_groups >= 2 &&
            (paced_scatter ||
             get_bool_from_env("FTRAIN_MOE_RS_A2A_SCHEDULED_SCATTER", true));
        if (adaptive_admission)
        {
            hipStream_t adaptive_stream = reduce_stream_;
            if (a2a_adaptive_admission_low_priority_)
            {
                FTRAIN_CHECK(a2a_scatter_stream_ != nullptr &&
                           a2a_scatter_done_event_ != nullptr);
                adaptive_stream = a2a_scatter_stream_;
                CUDA_CHECK(hipStreamWaitEvent(adaptive_stream,
                                              flags_reset_event_,
                                              0));
            }
            const int physical_n_tiles = ceil_div(n_dim_, 256);
            const int64_t counter_offset =
                static_cast<int64_t>(rank_tile_count) * physical_n_tiles;
            FTRAIN_CHECK(counter_offset + chunks.num_groups <=
                       v36_task_control_capacity_);
            moe_reduce_rs_topk_reduce_a2a_adaptive_admitted(
                gemm_output_.data_ptr(),
                reduce_scale,
                d_reduce_buffer_ptrs_,
                n_chunk_done_flags_.data_ptr<int>(),
                v36_task_control_.data_ptr<int>() + counter_offset,
                rank_tile_count,
                physical_n_tiles,
                ntokens,
                n_dim_,
                rank_,
                world_size_,
                topk_,
                chunks.chunk_size_n,
                chunks.num_groups,
                a2a_adaptive_admission_lead_groups_,
                a2a_adaptive_admission_lead_percent_,
                a2a_adaptive_admission_early_blocks_,
                a2a_adaptive_admission_final_blocks_,
                input_dtype_,
                adaptive_stream);
            if (a2a_adaptive_admission_low_priority_)
            {
                CUDA_CHECK(hipEventRecord(a2a_scatter_done_event_,
                                          adaptive_stream));
                CUDA_CHECK(hipStreamWaitEvent(reduce_stream_,
                                              a2a_scatter_done_event_,
                                              0));
            }
        }
        else if (admitted_scatter)
        {
            hipStream_t admitted_stream = reduce_stream_;
            if (a2a_admitted_scatter_low_priority_)
            {
                FTRAIN_CHECK(a2a_scatter_stream_ != nullptr &&
                           a2a_scatter_done_event_ != nullptr);
                admitted_stream = a2a_scatter_stream_;
                CUDA_CHECK(hipStreamWaitEvent(admitted_stream,
                                              flags_reset_event_,
                                              0));
            }
            moe_reduce_rs_topk_reduce_a2a_admitted(
                gemm_output_.data_ptr(),
                reduce_scale,
                d_reduce_buffer_ptrs_,
                n_chunk_done_flags_.data_ptr<int>(),
                ntokens,
                n_dim_,
                rank_,
                world_size_,
                topk_,
                chunks.chunk_size_n,
                chunks.num_groups,
                a2a_admitted_scatter_batch_groups_,
                a2a_admitted_scatter_early_blocks_,
                a2a_admitted_scatter_final_blocks_,
                input_dtype_,
                admitted_stream);
            if (a2a_admitted_scatter_low_priority_)
            {
                CUDA_CHECK(hipEventRecord(a2a_scatter_done_event_,
                                          admitted_stream));
                CUDA_CHECK(hipStreamWaitEvent(reduce_stream_,
                                              a2a_scatter_done_event_,
                                              0));
            }
        }
        else if (scheduled_scatter)
        {
            const int split_group = std::max(
                1,
                std::min(
                    chunks.num_groups - 1,
                    get_int_from_env(
                        "FTRAIN_MOE_RS_A2A_SCHEDULED_SCATTER_SPLIT_GROUP",
                        chunks.num_groups - 1)));
            const int early_budget = std::max(
                1,
                get_int_from_env(
                    "FTRAIN_MOE_RS_A2A_SCHEDULED_SCATTER_EARLY_BLOCKS",
                    16));
            const int late_budget = std::max(
                1,
                get_int_from_env(
                    "FTRAIN_MOE_RS_A2A_SCHEDULED_SCATTER_LATE_BLOCKS",
                    128));
            if (paced_scatter)
            {
                moe_reduce_rs_topk_reduce_a2a_paced_range(
                    gemm_output_.data_ptr(),
                    reduce_scale,
                    d_reduce_buffer_ptrs_,
                    n_chunk_done_flags_.data_ptr<int>(),
                    ntokens,
                    n_dim_,
                    rank_,
                    world_size_,
                    topk_,
                    chunks.chunk_size_n,
                    chunks.num_groups,
                    0,
                    split_group,
                    early_budget,
                    a2a_paced_scatter_check_iterations_,
                    a2a_paced_scatter_sleep_steps_,
                    input_dtype_,
                    reduce_stream_);
            }
            else
            {
                moe_reduce_rs_topk_reduce_a2a_range(
                    gemm_output_.data_ptr(),
                    reduce_scale,
                    d_reduce_buffer_ptrs_,
                    n_chunk_done_flags_.data_ptr<int>(),
                    ntokens,
                    n_dim_,
                    rank_,
                    world_size_,
                    topk_,
                    chunks.chunk_size_n,
                    0,
                    split_group,
                    early_budget,
                    input_dtype_,
                    reduce_stream_);
            }
            CUDA_CHECK(
                hipStreamWaitEvent(reduce_stream_, gemm_done_event_, 0));
            moe_reduce_rs_topk_reduce_a2a_range(
                gemm_output_.data_ptr(),
                reduce_scale,
                d_reduce_buffer_ptrs_,
                n_chunk_done_flags_.data_ptr<int>(),
                ntokens,
                n_dim_,
                rank_,
                world_size_,
                topk_,
                chunks.chunk_size_n,
                split_group,
                chunks.num_groups - split_group,
                late_budget,
                input_dtype_,
                reduce_stream_);
        }
        else
        {
            moe_reduce_rs_topk_reduce_a2a(
                gemm_output_.data_ptr(),
                reduce_scale,
                d_reduce_buffer_ptrs_,
                n_chunk_done_flags_.data_ptr<int>(),
                ntokens,
                n_dim_,
                rank_,
                world_size_,
                topk_,
                chunks.chunk_size_n,
                chunks.num_groups,
                input_dtype_,
                a2a_decoupled_scatter_enabled_ ? a2a_scatter_stream_
                                                : reduce_stream_);
        }
        if (a2a_decoupled_scatter_enabled_)
        {
            CUDA_CHECK(hipEventRecord(a2a_scatter_done_event_,
                                      a2a_scatter_stream_));
            CUDA_CHECK(hipStreamWaitEvent(reduce_stream_,
                                          a2a_scatter_done_event_,
                                          0));
        }
        sync_reduce_stream();
        moe_reduce_rs_sum_a2a(d_reduce_buffer_ptrs_,
                              output.data_ptr(),
                              ntokens / world_size_,
                              n_dim_,
                              rank_,
                              world_size_,
                              input_dtype_,
                              reduce_stream_);
        sync_reduce_stream();
    }



    Tensor run_impl(Tensor input,
                           Tensor weight,
                           Tensor chosen_experts,
                           Tensor expert_weight,
                           std::optional<Tensor> output,
                           bool defer_completion_wait)
    {
        FTRAIN_CHECK(!a2a_async_pending_)
            << "MoEReduceRS has a pending async A2A; call wait_a2a first";
        ScopedEnvironmentValues gemm_contract_env = make_gemm_contract_environment();
        active_weight_native_kn_ = gemm_variant_ == "rs_nn";
        validate_inputs(input, weight, chosen_experts, expert_weight);
        const int ntokens = chosen_experts.size(0);
        Tensor out = make_output(ntokens, std::move(output));
        const bool ragged_partition = has_ragged_partition(ntokens);
        // Normally an equal rank-M partition uses the lower-overhead fixed-M
        // A2A path.  Keep that default, but permit benchmark/diagnostic runs
        // to exercise the native ragged layout with equal rank-M metadata so
        // its overhead can be measured directly against the fixed-M route.
        const bool force_ragged_a2a_signal =
            !rank_m_sizes_.empty() &&
            get_bool_from_env("FTRAIN_MOE_RS_A2A_RAGGED_SIGNAL_FORCE", false);
        const bool ragged_a2a_layout =
            ragged_partition || force_ragged_a2a_signal;
        const bool route_ragged_signal =
            route_kind_ == MoERSRouteKind::A2ARaggedSignal;
        const bool ragged_a2a_requested =
            ragged_a2a_layout &&
            (route_ragged_signal ||
             get_bool_from_env("FTRAIN_MOE_RS_A2A_RAGGED_SIGNAL", true));
        const bool active_ragged_a2a_signal =
            ragged_a2a_requested && nnodes_ == 1 && world_size_ >= 2 &&
            // The native ragged signal launcher is specifically the
            // hardware-wait implementation and has no stream-wait variant,
            // so it keys off the device capability instead of the fixed-M
            // shape policy (which only decides between the two fixed-M
            // sub-paths).
            a2a_hardware_wait_capable_ && a2a_signal_pipeline_enabled_ &&
            a2a_chunks_.n_chunk_signal;
        if (ragged_a2a_requested && !active_ragged_a2a_signal &&
            (route_ragged_signal ||
             get_bool_from_env("FTRAIN_MOE_RS_A2A_RAGGED_SIGNAL_STRICT", false)))
        {
            FTRAIN_CHECK(false)
                << "Ragged A2A signal requires one node, world >= 2, a device "
                   "that supports device-address stream waits, and per-N-chunk "
                   "AG signals; the ProcessGroup fallback was removed";
        }
        // The ProcessGroup and ring transport routes were removed: every
        // remaining configuration must run the single-node A2A family, and
        // ragged input must be admissible for the native ragged signal path.
        FTRAIN_CHECK(nnodes_ == 1)
            << "MoEReduceRS now supports single-node configurations only "
               "(the ProcessGroup/ring routes were removed)";
        FTRAIN_CHECK(!ragged_a2a_layout || active_ragged_a2a_signal)
            << "Ragged input requires the native A2A signal route with "
               "hardware-wait enabled; the ProcessGroup fallback was removed";
        const ChunkConfig &chunks = a2a_chunks_;
        const bool n_chunk_signal = chunks.n_chunk_signal;
        last_chunk_size_n_ = chunks.chunk_size_n;
        last_num_groups_ = chunks.num_groups;
        last_n_chunk_signal_ = n_chunk_signal;
        const bool upstream_only =
            get_bool_from_env("FTRAIN_MOE_RS_UPSTREAM_ONLY", false);
        FTRAIN_CHECK(!upstream_only)
            << "FTRAIN_MOE_RS_UPSTREAM_ONLY is an A2A diagnostic only";
        const bool can_topk_stage =
            !upstream_only &&
            // The NVIDIA single-node path reduces/scales grouped-GEMM output
            // directly.  On gfx936 that path is also faster than materializing
            // the FP32 top-k stage for both A2A and host-stepped Ring.  Keep
            // top-k staging as the multi-node/ProcessGroup default because
            // ProcessGroup reduce-scatter consumes that locally reduced
            // tensor.  Single-node ragged signal A2A keeps the faster
            // scale-in-reduce path unless explicitly overridden.
            get_bool_from_env("FTRAIN_MOE_RS_EPILOGUE_TOPK_STAGE", false) &&
            input_dtype_ == ScalarType::BFloat16 && n_dim_ % 256 == 0 &&
            k_dim_ % 32 == 0 &&
            // A PG route may still use the top-k stage when AG chunk signals
            // are disabled: the host publishes all done flags after GEMM as
            // a serialized fallback.  The normal overlapped route requires
            // n_chunk_signal, but rejecting the fallback here would make a
            // valid multi-node configuration fail before reduction.
            (chunks.num_groups == 1 || chunks.n_chunk_signal) &&
            topk_stage_.defined();
        const bool scale_in_reduce = get_bool_from_env(
            "FTRAIN_MOE_RS_AG_RANK_TILE_SCALE_IN_REDUCE",
            !can_topk_stage);
        const bool topk_stage = can_topk_stage && !scale_in_reduce;
        const bool direct_scatter_override =
            !topk_stage &&
            (a2a_paced_scatter_enabled_ || a2a_admitted_scatter_enabled_ ||
             a2a_adaptive_admission_enabled_);
        const bool active_ragged_a2a_cooperative =
            active_ragged_a2a_signal && !direct_scatter_override &&
            a2a_cooperative_fused_enabled_ &&
            get_bool_from_env("FTRAIN_MOE_RS_A2A_RAGGED_COOPERATIVE_FUSED",
                              false);
        const bool active_a2a_cooperative_fused =
            !ragged_a2a_layout && !direct_scatter_override &&
            a2a_cooperative_fused_enabled_;
        const bool active_a2a_group_pipeline =
            active_a2a_cooperative_fused && a2a_group_pipeline_enabled_ &&
            chunks.n_chunk_signal && a2a_hip_cooperative_launch_enabled_;
        const bool active_a2a_signal_pipeline =
            (active_ragged_a2a_signal && !active_ragged_a2a_cooperative) ||
            (!ragged_a2a_layout && !direct_scatter_override &&
             a2a_signal_pipeline_enabled_ &&
             chunks.n_chunk_signal);

        const bool debug_route =
            rank_ == 0 && get_bool_from_env("FTRAIN_MOE_RS_DEBUG_ROUTE", false);
        if (debug_route)
        {
            std::printf("MoEReduceRS source dispatch=%s nnodes=%d "
                        "local_world=%d chunk_size_n=%d groups=%d "
                        "n_chunk_signal=%d topk_stage=%d "
                        "a2a_fused=%d a2a_group_pipeline=%d "
                        "a2a_hip_coop=%d "
                        "a2a_signal_pipeline=%d a2a_signal_stream_wait=%d "
                        "a2a_hardware_wait=%d a2a_split_pipeline=%d "
                        "producer_blocks=%d consumer_blocks=%d "
                        "a2a_signal_blocks=%d wave64_512_scatter=%d "
                        "wave64_512_sum=%d paced_scatter=%d "
                        "paced_check_iters=%d paced_sleep_steps=%d "
                        "admitted_scatter=%d admitted_batch_groups=%d "
                        "admitted_low_priority=%d "
                        "admitted_early_blocks=%d "
                        "admitted_final_blocks=%d "
                        "cu_partition=%d comm_cus=%d\n",
                        active_ragged_a2a_cooperative
                            ? "a2a_ragged_cooperative"
                            : (active_ragged_a2a_signal ? "a2a_ragged_signal"
                                                        : "a2a"),
                        nnodes_,
                        local_world_size_,
                        chunks.chunk_size_n,
                        chunks.num_groups,
                        static_cast<int>(n_chunk_signal),
                        static_cast<int>(topk_stage),
                        static_cast<int>(active_a2a_cooperative_fused ||
                                         active_ragged_a2a_cooperative),
                        static_cast<int>(active_a2a_group_pipeline),
                        static_cast<int>((active_a2a_cooperative_fused ||
                                          active_ragged_a2a_cooperative) &&
                                         a2a_hip_cooperative_launch_enabled_),
                        static_cast<int>(active_a2a_signal_pipeline),
                        static_cast<int>(active_a2a_signal_pipeline &&
                                         a2a_signal_stream_wait_enabled_),
                        static_cast<int>(active_a2a_signal_pipeline &&
                                         a2a_hardware_wait_enabled_),
                        static_cast<int>(active_a2a_signal_pipeline &&
                                         a2a_split_pipeline_enabled_),
                        a2a_signal_producer_block_budget_,
                        a2a_signal_consumer_block_budget_,
                        a2a_signal_block_budget_,
                        static_cast<int>(a2a_wave64_512_scatter_enabled_),
                        static_cast<int>(a2a_wave64_512_sum_enabled_),
                        static_cast<int>(a2a_paced_scatter_enabled_),
                        a2a_paced_scatter_check_iterations_,
                        a2a_paced_scatter_sleep_steps_,
                        static_cast<int>(a2a_admitted_scatter_enabled_),
                        a2a_admitted_scatter_batch_groups_,
                        static_cast<int>(a2a_admitted_scatter_low_priority_),
                        a2a_admitted_scatter_early_blocks_,
                        a2a_admitted_scatter_final_blocks_,
                        static_cast<int>(a2a_cu_partition_enabled_),
                        a2a_comm_cu_count_);
            std::printf(
                "MoEReduceRS staged_block_budget=%d early_producer_blocks=%d "
                "early_consumer_blocks=%d full_budget_start_group=%d\n",
                static_cast<int>(a2a_staged_block_budget_enabled_ &&
                                 !active_weight_native_kn_),
                a2a_signal_early_producer_block_budget_,
                a2a_signal_early_consumer_block_budget_,
                a2a_signal_full_budget_start_group_);
            if (a2a_adaptive_admission_enabled_)
            {
                std::printf(
                    "MoEReduceRS adaptive admission lead_groups=%d "
                    "lead_percent=%d early_blocks=%d final_blocks=%d "
                    "low_priority=%d workspace_words=%d\n",
                    a2a_adaptive_admission_lead_groups_,
                    a2a_adaptive_admission_lead_percent_,
                    a2a_adaptive_admission_early_blocks_,
                    a2a_adaptive_admission_final_blocks_,
                    static_cast<int>(a2a_adaptive_admission_low_priority_),
                    v36_task_control_capacity_);
            }
            std::fflush(stdout);
        }

        hipStream_t current_stream =
            context_->stream();

        const Tensor &gemm_weight = weight;
        CUDA_CHECK(hipEventRecord(input_ready_event_, current_stream));
        CUDA_CHECK(hipStreamWaitEvent(gemm_stream_, input_ready_event_, 0));
        CUDA_CHECK(hipStreamWaitEvent(reduce_stream_, input_ready_event_, 0));

        // The old host implementation reset the signal state on the GEMM
        // stream and published that reset before the reduce stream could
        // inspect the flags. Keeping the reset on current_stream can race a
        // later invocation and does not reproduce the old stream graph.
        CUDA_CHECK(hipMemsetAsync(n_chunk_done_flags_.data_ptr<int>(),
                                  0,
                                  sizeof(int) * chunks.num_groups,
                                  gemm_stream_));
        CUDA_CHECK(hipMemsetAsync(n_chunk_done_counts_.data_ptr<int>(),
                                  0,
                                  sizeof(int) * chunks.num_groups,
                                  gemm_stream_));
        CUDA_CHECK(hipMemsetAsync(gemm_done_flags_.data_ptr<int>(),
                                  0,
                                  sizeof(int) * chunks.num_groups,
                                  gemm_stream_));
        CUDA_CHECK(hipMemsetAsync(gemm_done_counts_.data_ptr<int>(),
                                  0,
                                  sizeof(int) * chunks.num_groups,
                                  gemm_stream_));
        if (v36_task_control_.defined())
        {
            CUDA_CHECK(hipMemsetAsync(v36_task_control_.data_ptr<int>(),
                                      0,
                                      sizeof(int) *
                                          static_cast<size_t>(
                                              v36_task_control_capacity_),
                                      gemm_stream_));
        }
        if (topk_stage)
        {
            CUDA_CHECK(hipMemsetAsync(topk_stage_.data_ptr(),
                                      0,
                                      sizeof(float) *
                                          static_cast<size_t>(ntokens) * n_dim_,
                                      gemm_stream_));
        }
        CUDA_CHECK(hipEventRecord(flags_reset_event_, gemm_stream_));
        CUDA_CHECK(hipStreamWaitEvent(reduce_stream_, flags_reset_event_, 0));

        const int rank_tile_count =
            prepare_metadata(chosen_experts, ntokens, gemm_stream_);
        launch_grouped_gemm(input,
                            gemm_weight,
                            chosen_experts,
                            expert_weight,
                            ntokens,
                            rank_tile_count,
                            chunks,
                            n_chunk_signal,
                            topk_stage);
        if (upstream_only)
        {
            // Diagnostic contract: expose the routed, unscaled grouped-GEMM
            // result and make the caller stream wait for its completion. This
            // isolates the current flat-input upstream (flag reset, cached
            // metadata/weight setup, direct-output GEMM, and done publication)
            // without launching any ReduceRS work. The normal operator path is
            // unchanged unless FTRAIN_MOE_RS_UPSTREAM_ONLY=1 is explicitly set.
            CUDA_CHECK(hipEventRecord(gemm_done_event_, gemm_stream_));
            CUDA_CHECK(hipStreamWaitEvent(current_stream, gemm_done_event_, 0));
            if (debug_route)
            {
                std::printf("MoEReduceRS source upstream-only "
                            "chunk_size_n=%d groups=%d rank_tiles=%d\n",
                            chunks.chunk_size_n,
                            chunks.num_groups,
                            rank_tile_count);
                std::fflush(stdout);
            }
            return gemm_output_.slice(0, 0, ntokens * topk_);
        }
        if (!topk_stage && !scale_in_reduce)
        {
            moe_reduce_rs_scale_output(gemm_output_.data_ptr(),
                                       expert_weight.data_ptr(),
                                       ntokens * topk_,
                                       n_dim_,
                                       input_dtype_,
                                       gemm_stream_);
        }
        if (!n_chunk_signal)
        {
            for (int group = 0; group < chunks.num_groups; ++group)
            {
                moe_reduce_rs_set_done_flags(
                    n_chunk_done_flags_.data_ptr<int>(), group, gemm_stream_);
            }
        }
        CUDA_CHECK(hipEventRecord(gemm_done_event_, gemm_stream_));

        // The reduction kernels consume the AG output through the per-group
        // done flags, so ordinary A2A/Ring does not need to make the caller
        // stream wait for the whole GEMM here.  The old host had this wait in
        // its PG handoff, but keeping it on current_stream would unnecessarily
        // block unrelated caller-stream work and erase part of the overlap.
        // PG routes add their narrower cast-ready waits in
        // launch_reduce_a2a_pg_rs below; the final reduce_done event remains
        // the operator's completion contract for every route.

        // The normal path relies on per-N-chunk device signals for overlap.
        // A separate post-GEMM scale, however, is not covered by those AG
        // signals: without this dependency the reduce kernel can consume a
        // chunk before moe_reduce_rs_scale_output has updated it.  The old ABI
        // exposes only a whole-output scale kernel, so this uncommon route
        // must serialize after the scale pass.  Scale-in-reduce and topk-stage
        // keep their N-chunk overlap.
        const bool needs_post_gemm_scale = !topk_stage && !scale_in_reduce;
        if (needs_post_gemm_scale ||
            get_bool_from_env("FTRAIN_MOE_RS_SYNC_BEFORE_REDUCE", false))
        {
            CUDA_CHECK(hipStreamWaitEvent(reduce_stream_, gemm_done_event_, 0));
        }

        if (active_ragged_a2a_cooperative)
        {
            launch_reduce_a2a_ragged_cooperative(out,
                                                  expert_weight,
                                                  ntokens,
                                                  chunks,
                                                  topk_stage,
                                                  scale_in_reduce);
        }
        else if (active_ragged_a2a_signal)
        {
            launch_reduce_a2a_ragged_signal(out,
                                            expert_weight,
                                            ntokens,
                                            chunks,
                                            topk_stage,
                                            scale_in_reduce);
        }
        else
        {
            launch_reduce_a2a(out,
                              expert_weight,
                              ntokens,
                              rank_tile_count,
                              chunks,
                              topk_stage,
                              scale_in_reduce);
        }
        CUDA_CHECK(hipEventRecord(reduce_done_event_, reduce_stream_));
        if (defer_completion_wait)
        {
            // The FC1 wgrad start records an event on current_stream.  Gate
            // that event on flag reset, not on full dgrad/A2A completion, so
            // wgrad can safely wait for freshly-produced chunk epochs without
            // observing the previous invocation's value of one.
            CUDA_CHECK(
                hipStreamWaitEvent(current_stream, flags_reset_event_, 0));
            pending_a2a_input_ = std::move(input);
            pending_a2a_weight_ = std::move(weight);
            pending_a2a_chosen_experts_ = std::move(chosen_experts);
            pending_a2a_expert_weight_ = std::move(expert_weight);
            pending_a2a_output_ = out;
            a2a_async_pending_ = true;
        }
        else
        {
            CUDA_CHECK(
                hipStreamWaitEvent(current_stream, reduce_done_event_, 0));
        }

        if (debug_route)
        {
            std::printf("MoEReduceRS source route=%s gemm_variant=%s "
                        "chunk_size_n=%d groups=%d "
                        "rank_tiles=%d n_chunk_signal=%d topk_stage=%d "
                        "scale_in_reduce=%d a2a_fused=%d "
                        "a2a_group_pipeline=%d "
                        "a2a_hip_coop=%d a2a_signal_pipeline=%d "
                        "a2a_signal_stream_wait=%d a2a_hardware_wait=%d "
                        "a2a_split_pipeline=%d producer_blocks=%d "
                        "consumer_blocks=%d a2a_signal_blocks=%d "
                        "wave64_512_scatter=%d wave64_512_sum=%d "
                        "paced_scatter=%d paced_check_iters=%d "
                        "paced_sleep_steps=%d admitted_scatter=%d "
                        "admitted_batch_groups=%d admitted_low_priority=%d "
                        "admitted_early_blocks=%d "
                        "admitted_final_blocks=%d "
                        "cu_partition=%d comm_cus=%d\n",
                        active_ragged_a2a_cooperative
                            ? "a2a_ragged_cooperative"
                            : (active_ragged_a2a_signal ? "a2a_ragged_signal"
                                                        : "a2a"),
                        gemm_variant_.c_str(),
                        chunks.chunk_size_n,
                        chunks.num_groups,
                        rank_tile_count,
                        static_cast<int>(n_chunk_signal),
                        static_cast<int>(topk_stage),
                        static_cast<int>(scale_in_reduce),
                        static_cast<int>(active_a2a_cooperative_fused ||
                                         active_ragged_a2a_cooperative),
                        static_cast<int>(active_a2a_group_pipeline),
                        static_cast<int>((active_a2a_cooperative_fused ||
                                          active_ragged_a2a_cooperative) &&
                                         a2a_hip_cooperative_launch_enabled_),
                        static_cast<int>(active_a2a_signal_pipeline),
                        static_cast<int>(active_a2a_signal_pipeline &&
                                         a2a_signal_stream_wait_enabled_),
                        static_cast<int>(active_a2a_signal_pipeline &&
                                         a2a_hardware_wait_enabled_),
                        static_cast<int>(active_a2a_signal_pipeline &&
                                         a2a_split_pipeline_enabled_),
                        a2a_signal_producer_block_budget_,
                        a2a_signal_consumer_block_budget_,
                        a2a_signal_block_budget_,
                        static_cast<int>(a2a_wave64_512_scatter_enabled_),
                        static_cast<int>(a2a_wave64_512_sum_enabled_),
                        static_cast<int>(a2a_paced_scatter_enabled_),
                        a2a_paced_scatter_check_iterations_,
                        a2a_paced_scatter_sleep_steps_,
                        static_cast<int>(a2a_admitted_scatter_enabled_),
                        a2a_admitted_scatter_batch_groups_,
                        static_cast<int>(a2a_admitted_scatter_low_priority_),
                        a2a_admitted_scatter_early_blocks_,
                        a2a_admitted_scatter_final_blocks_,
                        static_cast<int>(a2a_cu_partition_enabled_),
                        a2a_comm_cu_count_);
            std::fflush(stdout);
        }
        return out;
    }

    MoERSRouteKind route_kind_;
    std::string route_;
    std::string gemm_variant_;
    std::shared_ptr<Context> context_;
    int32_t nnodes_;
    int32_t max_ntokens_;
    int32_t n_dim_;
    int32_t k_dim_;
    int32_t num_experts_;
    int32_t topk_;
    ScalarType input_dtype_;
    int32_t n_chunks_;
    int32_t active_ntokens_ = 0;
    std::vector<int64_t> rank_m_sizes_;
    std::vector<int> rank_m_offsets_;
    int64_t reduce_buffer_rows_ = 0;
    int world_size_;
    int rank_;
    int local_world_size_;
    int local_rank_;
    int block_m_;
    int max_rank_tiles_;
    int source_sorted_capacity_;
    ChunkConfig a2a_chunks_;
    ChunkConfig ring_chunks_;
    int reduce_buffer_n_;
    int max_num_groups_;
    int ring_block_budget_;
    static constexpr int kA2AResidentBlocks = 32;
    bool a2a_split_pipeline_enabled_ = false;
    bool a2a_hardware_wait_enabled_ = false;
    // Device capability (hipDeviceAttributeCanUseStreamWaitValue) for the
    // hardware-wait kernels, independent of the fixed-M shape policy.
    bool a2a_hardware_wait_capable_ = false;
    bool a2a_signal_pipeline_enabled_ = false;
    bool a2a_signal_stream_wait_enabled_ = false;
    bool a2a_wave64_512_scatter_enabled_ = false;
    bool a2a_wave64_512_sum_enabled_ = false;
    int a2a_signal_block_budget_ = 16;
    int a2a_signal_producer_block_budget_ = 24;
    int a2a_signal_consumer_block_budget_ = 8;
    bool a2a_staged_block_budget_enabled_ = false;
    int a2a_signal_early_producer_block_budget_ = 16;
    int a2a_signal_early_consumer_block_budget_ = 8;
    int a2a_signal_full_budget_start_group_ = 0;
    bool a2a_cu_partition_enabled_ = false;
    int a2a_comm_cu_count_ = 16;
    bool a2a_group_pipeline_enabled_ = false;
    bool a2a_paced_scatter_enabled_ = false;
    int a2a_paced_scatter_check_iterations_ = 4;
    int a2a_paced_scatter_sleep_steps_ = 8;
    bool a2a_admitted_scatter_enabled_ = false;
    bool a2a_admitted_scatter_low_priority_ = true;
    int a2a_admitted_scatter_batch_groups_ = 1;
    int a2a_admitted_scatter_early_blocks_ = 16;
    int a2a_admitted_scatter_final_blocks_ = 128;
    bool a2a_adaptive_admission_enabled_ = false;
    bool a2a_adaptive_admission_low_priority_ = true;
    int a2a_adaptive_admission_lead_groups_ = 1;
    int a2a_adaptive_admission_lead_percent_ = 50;
    int a2a_adaptive_admission_early_blocks_ = 16;
    int a2a_adaptive_admission_final_blocks_ = 128;
    bool a2a_decoupled_scatter_enabled_ = false;
    bool active_weight_native_kn_ = false;
    int a2a_epoch_ = 0;
    int a2a_cooperative_epoch_ = 0;
    int a2a_signal_stride_ = 0;
    int a2a_signal_elems_ = 0;
    // Keep counter/remote-ready/local-ready in separate cache lines.  The
    // device still addresses the first three words as slots 0..2; the extra
    // words prevent adjacent (group, segment) pairs from false sharing on
    // system-scope IPC atomics.
    bool a2a_cooperative_fused_enabled_ = false;
    bool a2a_hip_cooperative_launch_enabled_ = false;
    Tensor gemm_output_;
    Tensor topk_stage_;
    Tensor output_buffer_;
    Tensor expert_counts_;
    Tensor expert_offsets_;
    Tensor expert_rank_bounds_;
    Tensor rank_tile_info_;
    Tensor metadata_scratch0_;
    Tensor sorted_index_workspace_;
    Tensor source_sorted_token_ids_;
    Tensor source_sorted_gather_index_;
    Tensor fc2_packed_input_;
    Tensor fc2_contract_info_;
    Tensor source_expert_rank_counts_;
    Tensor source_expert_rank_counters_;
    Tensor source_expert_rank_offsets_;
    Tensor source_global_expert_offsets_;
    Tensor source_rank_cursors_;
    Tensor source_rank_tile_info_;
    Tensor source_tile_experts_;
    Tensor source_tile_starts_;
    Tensor source_tile_rank_starts_;
    Tensor source_tile_rank_ends_;
    Tensor source_ntiles_;
    Tensor n_chunk_done_counts_;
    Tensor n_chunk_done_flags_;
    Tensor gemm_done_counts_;
    Tensor gemm_done_flags_;
    Tensor pending_a2a_input_;
    Tensor pending_a2a_weight_;
    Tensor pending_a2a_chosen_experts_;
    Tensor pending_a2a_expert_weight_;
    Tensor pending_a2a_output_;
    bool a2a_async_pending_ = false;
    int32_t last_chunk_size_n_ = 0;
    int32_t last_num_groups_ = 0;
    bool last_n_chunk_signal_ = false;
    Tensor v36_task_control_;
    int v36_task_control_capacity_ = 0;
    Tensor a2a_grid_barrier_;
    Tensor pg_reduce_buffer_;
    std::vector<Tensor> reduce_buffers_;
    std::vector<Tensor> a2a_signal_buffers_;
    std::vector<const void *> a2a_signal_host_ptrs_;
#ifndef FTRAIN_SHM_USE_NVSHMEM
    std::vector<Tensor> sync_buffers_;
#endif

    std::vector<int> host_expert_counts_;
    std::vector<int> host_expert_offsets_;
    std::vector<int> host_expert_rank_bounds_;
    std::vector<int> host_rank_tile_info_;

    // Metadata cache key mirrors the AG recovery shim: routing metadata is
    // reusable only for the same tensor storage, version, dtype, and token
    // count.  A version change (including an in-place copy_) invalidates it.
    bool metadata_cache_valid_ = false;
    Tensor metadata_topk_tensor_;
    const void *metadata_topk_src_ = nullptr;
    int64_t metadata_topk_version_ = -1;
    int metadata_topk_ntokens_ = 0;
    ScalarType metadata_topk_dtype_ = ScalarType::Undefined;
    int metadata_rank_tile_count_ = 0;

    void **d_reduce_buffer_ptrs_ = nullptr;
    void **d_a2a_signal_buffer_ptrs_ = nullptr;
    int *d_rank_m_offsets_ = nullptr;
    int **d_sync_buffer_ptrs_ = nullptr;
    hipStream_t gemm_stream_ = nullptr;
    hipStream_t reduce_stream_ = nullptr;
    hipStream_t a2a_scatter_stream_ = nullptr;
    hipEvent_t input_ready_event_ = nullptr;
    hipEvent_t flags_reset_event_ = nullptr;
    hipEvent_t gemm_done_event_ = nullptr;
    hipEvent_t reduce_done_event_ = nullptr;
    hipEvent_t a2a_scatter_done_event_ = nullptr;
};

MoEReduceRS::MoEReduceRS(std::shared_ptr<Context> context,
                         int32_t nnodes,
                         int32_t max_ntokens,
                         int32_t n_dim,
                         int32_t k_dim,
                         int32_t num_experts,
                         int32_t topk,
                        ScalarType input_dtype,
                        int32_t n_chunks,
                        const std::string &route,
                        const std::string &gemm_variant)
    : pimpl_(std::make_unique<MoEReduceRSImpl>(context,
                                               nnodes,
                                               max_ntokens,
                                               n_dim,
                                               k_dim,
                                               num_experts,
                                               topk,
                                               input_dtype,
                                               n_chunks,
                                               route,
                                               gemm_variant))
{
}

MoEReduceRS::~MoEReduceRS() = default;

void MoEReduceRS::set_rank_m_sizes(
    int32_t active_ntokens,
    std::vector<int64_t> rank_m_sizes)
{
    pimpl_->set_rank_m_sizes(active_ntokens, std::move(rank_m_sizes));
}

Tensor MoEReduceRS::run_a2a(Tensor input,
                                   Tensor weight,
                                   Tensor chosen_experts,
                                   Tensor expert_weight,
                                   std::optional<Tensor> output)
{
    return pimpl_->run_a2a(std::move(input),
                           std::move(weight),
                           std::move(chosen_experts),
                           std::move(expert_weight),
                           std::move(output));
}

Tensor MoEReduceRS::run_a2a_async(
    Tensor input,
    Tensor weight,
    Tensor chosen_experts,
    Tensor expert_weight,
    std::optional<Tensor> output)
{
    return pimpl_->run_a2a_async(std::move(input),
                                 std::move(weight),
                                 std::move(chosen_experts),
                                 std::move(expert_weight),
                                 std::move(output));
}

void MoEReduceRS::wait_a2a()
{
    pimpl_->wait_a2a();
}

Tensor MoEReduceRS::n_chunk_done_flags()
{
    return pimpl_->n_chunk_done_flags();
}

int32_t MoEReduceRS::last_chunk_size_n() const
{
    return pimpl_->last_chunk_size_n();
}

int32_t MoEReduceRS::last_num_groups() const
{
    return pimpl_->last_num_groups();
}

bool MoEReduceRS::last_n_chunk_signal() const
{
    return pimpl_->last_n_chunk_signal();
}

}
} // namespace moe
} // namespace ftrain
