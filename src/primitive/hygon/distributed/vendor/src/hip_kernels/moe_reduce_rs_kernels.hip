#include "hygon_vendor.h"
#include "hip/hip_bfloat16.h"
#include "hip/hip_cooperative_groups.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "hip_kernels/moe_reduce_rs_kernels.h"
#include <c10/core/ScalarType.h>
#include <algorithm>
#include <cstdlib>
#include <stdint.h>

namespace bytedance
{
namespace hygon_vendor
{
namespace
{

template <typename T>
__device__ inline float load_as_float(const T &value);

template <>
__device__ inline float load_as_float<__half>(const __half &value)
{
    return __half2float(value);
}

template <>
__device__ inline float load_as_float<__hip_bfloat16>(
    const __hip_bfloat16 &value)
{
    return __bfloat162float(value);
}

template <>
__device__ inline float load_as_float<float>(const float &value)
{
    return value;
}

template <typename T>
__device__ inline T store_from_float(float value);

template <>
__device__ inline __half store_from_float<__half>(float value)
{
    return __float2half(value);
}

template <>
__device__ inline __hip_bfloat16 store_from_float<__hip_bfloat16>(float value)
{
    return __float2bfloat16(value);
}

template <>
__device__ inline float store_from_float<float>(float value)
{
    return value;
}

template <typename IndexT>
__device__ inline int load_index(const void *ptr, int offset)
{
    return static_cast<int>(static_cast<const IndexT *>(ptr)[offset]);
}

template <typename T, typename IndexT>
__global__ void moe_reduce_rs_compute_kernel(const T *input,
                                             const T *weight,
                                             const void *chosen_experts,
                                             const T *expert_weight,
                                             T *partial,
                                             int ntokens,
                                             int n_dim,
                                             int k_per_rank,
                                             int num_experts,
                                             int topk)
{
    const int64_t total = static_cast<int64_t>(ntokens) * n_dim;
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total)
    {
        return;
    }

    const int n = linear % n_dim;
    const int token = linear / n_dim;
    float acc = 0.0f;
    for (int tk = 0; tk < topk; ++tk)
    {
        const int flat_row = token * topk + tk;
        const int expert = load_index<IndexT>(chosen_experts, flat_row);
        if (expert < 0 || expert >= num_experts)
        {
            continue;
        }
        float gemm = 0.0f;
        const T *x = input + static_cast<int64_t>(flat_row) * k_per_rank;
        const T *w = weight +
                     (static_cast<int64_t>(expert) * k_per_rank * n_dim) + n;
        for (int k = 0; k < k_per_rank; ++k)
        {
            gemm += load_as_float<T>(x[k]) *
                    load_as_float<T>(w[static_cast<int64_t>(k) * n_dim]);
        }
        acc += gemm * load_as_float<T>(expert_weight[flat_row]);
    }
    partial[linear] = store_from_float<T>(acc);
}

template <typename T>
__global__ void moe_reduce_rs_sum_kernel(void **partial_ptrs,
                                         T *output,
                                         int world_size,
                                         int rank,
                                         int ntokens,
                                         int n_dim)
{
    const int tokens_per_rank = ntokens / world_size;
    const int64_t total = static_cast<int64_t>(tokens_per_rank) * n_dim;
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total)
    {
        return;
    }

    const int n = linear % n_dim;
    const int local_token = linear / n_dim;
    const int global_token = rank * tokens_per_rank + local_token;
    float acc = 0.0f;
    for (int peer = 0; peer < world_size; ++peer)
    {
        const T *partial = static_cast<const T *>(partial_ptrs[peer]);
        acc += load_as_float<T>(
            partial[static_cast<int64_t>(global_token) * n_dim + n]);
    }
    output[linear] = store_from_float<T>(acc);
}

// The reconstructed MoEReduceRS host deliberately launches the reduce side
// before the rank-tile GEMM has completed.  Keep the waiting grids small so a
// high-priority reduce stream cannot occupy the whole device while the normal
// priority GEMM is still producing the requested N chunk.
constexpr int kReduceThreads = 256;
// Keep 32 as the generic launch cap and non-group-pipeline fallback.  The
// per-group standard-[K,N] path applies its smaller measured block budget in
// launch_a2a_cooperative_fused_impl so its resident scatter does not starve
// the bandwidth-sensitive grouped GEMM.
constexpr int kReduceBlocks = 32;
// The NVIDIA persistent Ring uses six very large (32-warp) programs.  This
// source kernel is the HCU-style host-stepped variant with 256-thread blocks;
// legacy/host-stepped launches remain capped at 32 blocks.  The all-groups
// wavefront route has a separate host-selected budget (safe default 32), so
// target shapes can use 48 blocks without changing the fallback schedule.
constexpr int kRingBlocks = 32;
constexpr int kWorkBlocks = 256;

__device__ inline void wait_for_done_flag(const int *done_flags, int group)
{
    if (threadIdx.x == 0)
    {
        const int *flag = done_flags + group;
        while (__atomic_load_n(flag, __ATOMIC_ACQUIRE) < 1)
        {
        }
    }
    __syncthreads();
}

// Persistent wave64/512x ragged producer: one launch covers every N group and
// spins on the AG done flags in-kernel, so the stream-wait scheduling can keep
// the wave64 fast path on the scatter side.  The launcher only selects this
// kernel when every group's chunk is a >=512 multiple of 512.


template <typename IndexT>
__global__ void prepare_counts_kernel(const IndexT *chosen_experts,
                                      int *expert_counts,
                                      int total_routes,
                                      int num_experts)
{
    for (int route = blockIdx.x * blockDim.x + threadIdx.x;
         route < total_routes;
         route += blockDim.x * gridDim.x)
    {
        const int expert = static_cast<int>(chosen_experts[route]);
        if (expert >= 0 && expert < num_experts)
        {
            atomicAdd(expert_counts + expert, 1);
        }
    }
}

__global__ void set_done_flag_kernel(int *done_flags, int group)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        __threadfence();
        atomicExch(done_flags + group, 1);
    }
}

template <typename T>
__global__ void scale_output_kernel(T *output,
                                    const T *scale,
                                    int rows,
                                    int n_dim)
{
    const int64_t total = static_cast<int64_t>(rows) * n_dim;
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int row = static_cast<int>(linear / n_dim);
        output[linear] = store_from_float<T>(
            load_as_float<T>(output[linear]) * load_as_float<T>(scale[row]));
    }
}

template <typename T>
__global__ void build_topk_stage_chunked_kernel(
    const T *__restrict__ gemm_output,
    const T *__restrict__ expert_weight,
    float *__restrict__ topk_stage,
    const int *__restrict__ gemm_done_flags,
    int *__restrict__ n_chunk_done_counts,
    int *__restrict__ n_chunk_done_flags,
    int ntokens,
    int n_dim,
    int topk,
    int chunk_size_n,
    int signal_value)
{
    const int group = blockIdx.y;
    if (gemm_done_flags != nullptr)
    {
        wait_for_done_flag(gemm_done_flags, group);
    }
    const int n_start = group * chunk_size_n;
    const int n_end = min(n_start + chunk_size_n, n_dim);
    const int group_cols = n_end - n_start;
    const int64_t group_total =
        static_cast<int64_t>(ntokens) * group_cols;
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < group_total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / group_cols);
        const int n = n_start + static_cast<int>(linear % group_cols);
        float value = 0.0f;
#pragma unroll
        for (int tk = 0; tk < 8; ++tk)
        {
            if (tk >= topk)
            {
                break;
            }
            const int row = token * topk + tk;
            value += load_as_float<T>(
                         gemm_output[static_cast<int64_t>(row) * n_dim + n]) *
                     load_as_float<T>(expert_weight[row]);
        }
        topk_stage[static_cast<int64_t>(token) * n_dim + n] = value;
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        const int done = atomicAdd(n_chunk_done_counts + group, 1) + 1;
        if (done == static_cast<int>(gridDim.x))
        {
            __threadfence();
            n_chunk_done_flags[group] = signal_value;
        }
    }
}


template <typename T, bool ApplyScale>
__device__ inline float reduce_topk_value(const T *gemm_output,
                                          const T *expert_weight,
                                          int token,
                                          int n,
                                          int n_dim,
                                          int topk)
{
    float value = 0.0f;
    const int first_row = token * topk;
    for (int tk = 0; tk < topk; ++tk)
    {
        const int row = first_row + tk;
        float term = load_as_float<T>(
            gemm_output[static_cast<int64_t>(row) * n_dim + n]);
        if (ApplyScale)
        {
            term *= load_as_float<T>(expert_weight[row]);
        }
        value += term;
    }
    return value;
}

template <typename T>
struct alignas(16) ValuePack8
{
    T values[8];
};

struct FloatPack8
{
    float values[8];
};

template <typename T>
__device__ inline ValuePack8<T> cast_pack8(const FloatPack8 &input)
{
    ValuePack8<T> output;
#pragma unroll
    for (int lane = 0; lane < 8; ++lane)
    {
        output.values[lane] = store_from_float<T>(input.values[lane]);
    }
    return output;
}


template <typename T, bool ApplyScale>
__device__ inline FloatPack8 reduce_topk_pack8(
    const T *gemm_output,
    const T *expert_weight,
    int token,
    int n,
    int n_dim,
    int topk)
{
    FloatPack8 output{{0.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 0.0f}};
    const int first_row = token * topk;
    for (int tk = 0; tk < topk; ++tk)
    {
        const int row = first_row + tk;
        const ValuePack8<T> input =
            *reinterpret_cast<const ValuePack8<T> *>(
                gemm_output + static_cast<int64_t>(row) * n_dim + n);
        float scale = 1.0f;
        if (ApplyScale)
        {
            scale = load_as_float<T>(expert_weight[row]);
        }
#pragma unroll
        for (int lane = 0; lane < 8; ++lane)
        {
            output.values[lane] +=
                load_as_float<T>(input.values[lane]) * scale;
        }
    }
    return output;
}

// Keep the common direct-output topk=6 reduction compile-time specialized.
// The fixed-M wave64 producer already gets this specialization by passing a
// literal 6 to reduce_topk_pack8.  The ragged producer receives topk as an
// API argument, so without a separate helper its inner reduction remains a
// runtime-counted loop even when the caller uses topk=6.
template <typename T, bool ApplyScale, int TopK>
__device__ inline FloatPack8 reduce_topk_static_pack8(
    const T *gemm_output,
    const T *expert_weight,
    int token,
    int n,
    int n_dim)
{
    static_assert(TopK > 0, "TopK must be a positive compile-time constant");
    FloatPack8 output{{0.0f, 0.0f, 0.0f, 0.0f,
                       0.0f, 0.0f, 0.0f, 0.0f}};
    const int first_row = token * TopK;
#pragma unroll
    for (int tk = 0; tk < TopK; ++tk)
    {
        const int row = first_row + tk;
        const ValuePack8<T> input =
            *reinterpret_cast<const ValuePack8<T> *>(
                gemm_output + static_cast<int64_t>(row) * n_dim + n);
        float scale = 1.0f;
        if (ApplyScale)
        {
            scale = load_as_float<T>(expert_weight[row]);
        }
#pragma unroll
        for (int lane = 0; lane < 8; ++lane)
        {
            output.values[lane] +=
                load_as_float<T>(input.values[lane]) * scale;
        }
    }
    return output;
}

template <typename T, int TopK>
__global__ void te_moe_unpermute_topk_reduce_pack8_kernel(
    const T *__restrict__ packed_output,
    const int32_t *__restrict__ inverse_sorted_rows,
    const T *__restrict__ expert_weight,
    T *__restrict__ partial,
    int ntokens,
    int n_dim,
    int runtime_topk)
{
    const int topk = TopK > 0 ? TopK : runtime_topk;
    const int packs_per_row = n_dim / 8;
    const int64_t total = static_cast<int64_t>(ntokens) * packs_per_row;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / packs_per_row);
        const int pack = static_cast<int>(linear % packs_per_row);
        const int route_base = token * topk;
        FloatPack8 accum{{0.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int tk = 0; tk < topk; ++tk)
        {
            const int route = route_base + tk;
            const int packed_row = inverse_sorted_rows[route];
            const ValuePack8<T> values =
                reinterpret_cast<const ValuePack8<T> *>(
                    packed_output + static_cast<int64_t>(packed_row) * n_dim)
                    [pack];
            const float scale = load_as_float<T>(expert_weight[route]);
#pragma unroll
            for (int lane = 0; lane < 8; ++lane)
            {
                accum.values[lane] +=
                    load_as_float<T>(values.values[lane]) * scale;
            }
        }
        reinterpret_cast<ValuePack8<T> *>(
            partial + static_cast<int64_t>(token) * n_dim)[pack] =
            cast_pack8<T>(accum);
    }
}

template <typename T, int TopK>
__global__ void te_moe_unpermute_topk_reduce_scalar_kernel(
    const T *__restrict__ packed_output,
    const int32_t *__restrict__ inverse_sorted_rows,
    const T *__restrict__ expert_weight,
    T *__restrict__ partial,
    int ntokens,
    int n_dim,
    int runtime_topk)
{
    const int topk = TopK > 0 ? TopK : runtime_topk;
    const int64_t total = static_cast<int64_t>(ntokens) * n_dim;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_dim);
        const int n = static_cast<int>(linear % n_dim);
        const int route_base = token * topk;
        float accum = 0.0f;
#pragma unroll
        for (int tk = 0; tk < topk; ++tk)
        {
            const int route = route_base + tk;
            const int packed_row = inverse_sorted_rows[route];
            accum += load_as_float<T>(
                         packed_output[static_cast<int64_t>(packed_row) *
                                           n_dim +
                                       n]) *
                     load_as_float<T>(expert_weight[route]);
        }
        partial[linear] = store_from_float<T>(accum);
    }
}

template <typename T, bool ApplyScale>
__global__ void topk_scatter_a2a_groups_kernel(
    const T *gemm_output,
    const T *expert_weight,
    void **reduce_buffer_ptrs,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int first_group,
    int group_count)
{
    const int ntokens_per_rank = ntokens / world_size;
    const int group_end = first_group + group_count;
    for (int group = first_group; group < group_end; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        int this_chunk_size_n = n_dim - chunk_offset_n;
        if (this_chunk_size_n > chunk_size_n)
        {
            this_chunk_size_n = chunk_size_n;
        }
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const bool use_pack8 = (n_dim & 7) == 0 &&
                               (chunk_offset_n & 7) == 0 &&
                               (this_chunk_size_n & 7) == 0;
        const int n_items =
            use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
        const int64_t total = static_cast<int64_t>(ntokens) * n_items;
        for (int64_t linear =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int token = static_cast<int>(linear / n_items);
            const int n_item = static_cast<int>(linear % n_items);
            const int n =
                chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
            const int peer = token / ntokens_per_rank;
            const int local_token = token - peer * ntokens_per_rank;
            T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            const int dst_row = rank * ntokens_per_rank + local_token;
            T *destination =
                peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;
            if (use_pack8)
            {
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                        gemm_output,
                        expert_weight,
                        token,
                        n,
                        n_dim,
                        topk));
            }
            else
            {
                *destination =
                    store_from_float<T>(reduce_topk_value<T, ApplyScale>(
                        gemm_output,
                        expert_weight,
                        token,
                        n,
                        n_dim,
                        topk));
            }
        }
    }
}

template <typename T, bool ApplyScale>
__global__ void topk_scatter_a2a_groups_wave64_512x_kernel(
    const T *gemm_output,
    const T *expert_weight,
    void **reduce_buffer_ptrs,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int topk,
    int chunk_size_n,
    int first_group,
    int group_count)
{
    constexpr int kWorldSize = 8;
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kSubtileSizeN = kWaveSize * 8;

    const int ntokens_per_rank = ntokens / kWorldSize;
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int first_token = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const int dst_row_base = rank * ntokens_per_rank;
    const int group_end = first_group + group_count;

    for (int group = first_group; group < group_end; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        const int this_chunk_size_n = min(chunk_size_n, n_dim - chunk_offset_n);
        const int num_subtiles = this_chunk_size_n / kSubtileSizeN;
        for (int token = first_token; token < ntokens; token += token_stride)
        {
            const int peer = token / ntokens_per_rank;
            const int local_token = token - peer * ntokens_per_rank;
            T *const peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            T *const destination_row = peer_buffer +
                static_cast<int64_t>(dst_row_base + local_token) * n_dim;
            for (int subtile = 0; subtile < num_subtiles; ++subtile)
            {
                const int n = chunk_offset_n + subtile * kSubtileSizeN +
                              lane * 8;
                *reinterpret_cast<ValuePack8<T> *>(destination_row + n) =
                    cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(gemm_output,
                                                                   expert_weight,
                                                                   token,
                                                                   n,
                                                                   n_dim,
                                                                   topk));
            }
        }
    }
}

__device__ inline void paced_scatter_backoff(const int *done_flags,
                                              int next_group,
                                              int num_groups,
                                              int sleep_steps)
{
    if (next_group >= num_groups || sleep_steps <= 0)
    {
        return;
    }

    // One acquire per wave is enough to observe producer progress.  All lanes
    // take the same branch, so s_sleep lowers VMEM/peer traffic duty cycle
    // without introducing a block-wide barrier in the grid-stride loop.
    int producer_ahead = 0;
    if ((threadIdx.x & 63) == 0)
    {
        producer_ahead =
            __atomic_load_n(done_flags + next_group, __ATOMIC_ACQUIRE) >= 1;
    }
    producer_ahead = __shfl(producer_ahead, 0, 64);
    if (!producer_ahead)
    {
        for (int step = 0; step < sleep_steps; ++step)
        {
            __builtin_amdgcn_s_sleep(1);
        }
    }
}

template <typename T, bool ApplyScale>
__global__ void topk_scatter_a2a_paced_groups_kernel(
    const T *gemm_output,
    const T *expert_weight,
    void **reduce_buffer_ptrs,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int first_group,
    int group_count,
    int check_iterations,
    int sleep_steps)
{
    const int ntokens_per_rank = ntokens / world_size;
    const int group_end = first_group + group_count;
    for (int group = first_group; group < group_end; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        const int this_chunk_size_n = min(chunk_size_n, n_dim - chunk_offset_n);
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const bool use_pack8 = (n_dim & 7) == 0 &&
                               (chunk_offset_n & 7) == 0 &&
                               (this_chunk_size_n & 7) == 0;
        const int n_items =
            use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
        const int64_t total = static_cast<int64_t>(ntokens) * n_items;
        int iteration = 0;
        for (int64_t linear =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int token = static_cast<int>(linear / n_items);
            const int n_item = static_cast<int>(linear % n_items);
            const int n =
                chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
            const int peer = token / ntokens_per_rank;
            const int local_token = token - peer * ntokens_per_rank;
            T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            const int dst_row = rank * ntokens_per_rank + local_token;
            T *destination =
                peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;
            if (use_pack8)
            {
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(gemm_output,
                                                                   expert_weight,
                                                                   token,
                                                                   n,
                                                                   n_dim,
                                                                   topk));
            }
            else
            {
                *destination = store_from_float<T>(
                    reduce_topk_value<T, ApplyScale>(gemm_output,
                                                     expert_weight,
                                                     token,
                                                     n,
                                                     n_dim,
                                                     topk));
            }

            ++iteration;
            if (iteration == check_iterations)
            {
                paced_scatter_backoff(
                    done_flags, group + 1, num_groups, sleep_steps);
                iteration = 0;
            }
        }
    }
}

template <typename T, bool ApplyScale>
__global__ void topk_scatter_a2a_chunk_kernel(
    const T *gemm_output,
    const T *expert_weight,
    void **reduce_buffer_ptrs,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int group,
    int chunk_offset_n,
    int chunk_size_n)
{
    wait_for_done_flag(done_flags, group);
    const int ntokens_per_rank = ntokens / world_size;
    const bool use_pack8 = (n_dim & 7) == 0 &&
                           (chunk_offset_n & 7) == 0 &&
                           (chunk_size_n & 7) == 0;
    const int n_items = use_pack8 ? chunk_size_n / 8 : chunk_size_n;
    const int64_t total = static_cast<int64_t>(ntokens) * n_items;
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_items);
        const int n_item = static_cast<int>(linear % n_items);
        const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
        const int peer = token / ntokens_per_rank;
        const int local_token = token - peer * ntokens_per_rank;
        T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
        const int dst_row = rank * ntokens_per_rank + local_token;
        T *destination =
            peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;
        if (use_pack8)
        {
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(gemm_output,
                                                               expert_weight,
                                                               token,
                                                               n,
                                                               n_dim,
                                                               topk));
        }
        else
        {
            *destination = store_from_float<T>(
                reduce_topk_value<T, ApplyScale>(gemm_output,
                                                 expert_weight,
                                                 token,
                                                 n,
                                                 n_dim,
                                                 topk));
        }
    }
}

template <typename T>
__global__ void topk_stage_scatter_a2a_kernel(
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int chunk_size_n,
    int num_groups)
{
    const int ntokens_per_rank = ntokens / world_size;
    for (int group = 0; group < num_groups; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        int this_chunk_size_n = n_dim - chunk_offset_n;
        if (this_chunk_size_n > chunk_size_n)
        {
            this_chunk_size_n = chunk_size_n;
        }
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const bool use_pack8 = (n_dim & 7) == 0 &&
                               (chunk_offset_n & 7) == 0 &&
                               (this_chunk_size_n & 7) == 0;
        const int n_items =
            use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
        const int64_t total = static_cast<int64_t>(ntokens) * n_items;
        for (int64_t linear =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int token = static_cast<int>(linear / n_items);
            const int n_item = static_cast<int>(linear % n_items);
            const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
            const int peer = token / ntokens_per_rank;
            const int local_token = token - peer * ntokens_per_rank;
            T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            const int dst_row = rank * ntokens_per_rank + local_token;
            T *destination =
                peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;
            const float *source =
                topk_stage + static_cast<int64_t>(token) * n_dim + n;
            if (use_pack8)
            {
                const float4 first =
                    *reinterpret_cast<const float4 *>(source);
                const float4 second =
                    *reinterpret_cast<const float4 *>(source + 4);
                ValuePack8<T> packed;
                packed.values[0] = store_from_float<T>(first.x);
                packed.values[1] = store_from_float<T>(first.y);
                packed.values[2] = store_from_float<T>(first.z);
                packed.values[3] = store_from_float<T>(first.w);
                packed.values[4] = store_from_float<T>(second.x);
                packed.values[5] = store_from_float<T>(second.y);
                packed.values[6] = store_from_float<T>(second.z);
                packed.values[7] = store_from_float<T>(second.w);
                *reinterpret_cast<ValuePack8<T> *>(destination) = packed;
            }
            else
            {
                *destination = store_from_float<T>(*source);
            }
        }
    }
}

// The formal path uses a cooperative grid group.  The retained software
// fallback is safe only when every block can be resident at once, so the host
// enables that fallback only on a device with at least kReduceBlocks CUs and
// this kernel remains capped to the staged A2A route's <= 32-block grid.
template <bool HipCooperativeLaunch>
__device__ inline void a2a_grid_barrier(int *state)
{
    if constexpr (HipCooperativeLaunch)
    {
        cooperative_groups::this_grid().sync();
        return;
    }

    __shared__ int observed_epoch;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        observed_epoch = __atomic_load_n(state + 1, __ATOMIC_ACQUIRE);
        const int arrival = atomicAdd(state, 1);
        if (arrival + 1 == static_cast<int>(gridDim.x))
        {
            // Publish every block's work before releasing this epoch.  The
            // peer-write phase performs a system fence in every producer
            // thread before entering this barrier.
            __threadfence();
            atomicExch(state, 0);
            __threadfence();
            atomicAdd(state + 1, 1);
        }
        else
        {
            while (__atomic_load_n(state + 1, __ATOMIC_ACQUIRE) ==
                   observed_epoch)
            {
            }
        }
    }
    __syncthreads();
}

__device__ inline void a2a_ipc_barrier(int **sync_buffer_ptrs,
                                       int rank,
                                       int world_size)
{
    if (blockIdx.x == 0 && threadIdx.x < world_size)
    {
        // Match cudaipc_barrier_all.hip exactly: publish this rank to the
        // selected peer, then consume the selected peer's publication in the
        // local rank-owned allocation.  Parallel lanes are important here;
        // they avoid imposing a rank-ordered dependency chain.
        int *publish = sync_buffer_ptrs[threadIdx.x] + rank;
        while (atomicCAS_system(publish, 0, 1) != 0)
        {
        }
        int *wait = sync_buffer_ptrs[rank] + threadIdx.x;
        while (atomicCAS_system(wait, 1, 0) != 1)
        {
        }
    }
    if (blockIdx.x == 0)
    {
        __syncthreads();
    }
}

template <typename T>
__device__ inline void a2a_sum_group(const T *local_buffer,
                                     T *output,
                                     int ntokens_per_rank,
                                     int n_dim,
                                     int world_size,
                                     int group,
                                     int chunk_size_n);

template <typename T,
          bool ApplyScale,
          bool UseTopkStage,
          bool HipCooperativeLaunch,
          bool Ragged>
__global__ void a2a_cooperative_fused_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    int **sync_buffer_ptrs,
    int *grid_barrier,
    const int *done_flags,
    T *output,
    const int *rank_m_offsets,
    int local_rows,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int epoch,
    bool group_pipeline)
{
    // In the ragged layout each destination buffer contains contiguous
    // [producer_rank][destination_local_row] segments.  Fixed-M retains its
    // division-only mapping and therefore does not pay for the prefix lookup.
    const int ntokens_per_rank =
        Ragged ? local_rows : ntokens / world_size;

    // Scatter every ready N group.  Keeping the loop in one resident grid
    // preserves GEMM/reduce overlap without one launch and two host barriers
    // per group.
    for (int group = 0; group < num_groups; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        int this_chunk_size_n = n_dim - chunk_offset_n;
        if (this_chunk_size_n > chunk_size_n)
        {
            this_chunk_size_n = chunk_size_n;
        }
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const bool use_pack8 = (n_dim & 7) == 0 &&
                               (chunk_offset_n & 7) == 0 &&
                               (this_chunk_size_n & 7) == 0;
        const int n_items =
            use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
        const int64_t total = static_cast<int64_t>(ntokens) * n_items;
        for (int64_t linear =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int token = static_cast<int>(linear / n_items);
            const int n_item = static_cast<int>(linear % n_items);
            const int n =
                chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
            int peer = 0;
            int local_token = 0;
            int peer_rows = ntokens_per_rank;
            if constexpr (Ragged)
            {
                while (peer + 1 < world_size &&
                       token >= rank_m_offsets[peer + 1])
                {
                    ++peer;
                }
                local_token = token - rank_m_offsets[peer];
                peer_rows =
                    rank_m_offsets[peer + 1] - rank_m_offsets[peer];
            }
            else
            {
                peer = token / ntokens_per_rank;
                local_token = token - peer * ntokens_per_rank;
            }
            T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            const int dst_row = rank * peer_rows + local_token;
            T *destination = peer_buffer +
                static_cast<int64_t>(dst_row) * n_dim + n;

            if (UseTopkStage)
            {
                const float *source =
                    topk_stage + static_cast<int64_t>(token) * n_dim + n;
                if (use_pack8)
                {
                    const float4 first =
                        *reinterpret_cast<const float4 *>(source);
                    const float4 second =
                        *reinterpret_cast<const float4 *>(source + 4);
                    ValuePack8<T> packed;
                    packed.values[0] = store_from_float<T>(first.x);
                    packed.values[1] = store_from_float<T>(first.y);
                    packed.values[2] = store_from_float<T>(first.z);
                    packed.values[3] = store_from_float<T>(first.w);
                    packed.values[4] = store_from_float<T>(second.x);
                    packed.values[5] = store_from_float<T>(second.y);
                    packed.values[6] = store_from_float<T>(second.z);
                    packed.values[7] = store_from_float<T>(second.w);
                    *reinterpret_cast<ValuePack8<T> *>(destination) = packed;
                }
                else
                {
                    *destination = store_from_float<T>(*source);
                }
            }
            else if (use_pack8)
            {
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                        gemm_output,
                        expert_weight,
                        token,
                        n,
                        n_dim,
                        topk));
            }
            else
            {
                *destination =
                    store_from_float<T>(reduce_topk_value<T, ApplyScale>(
                        gemm_output,
                        expert_weight,
                        token,
                        n,
                        n_dim,
                        topk));
            }
        }

        if (group_pipeline)
        {
            // The group-pipeline consumer is only defined for the uniform
            // source stride. The host rejects it for ragged cooperative A2A.
            if constexpr (Ragged)
            {
                return;
            }
            else
            {
            // A per-group cooperative hand-off keeps the kernel resident but
            // lets the next GEMM N-group run while this group is being
            // reduced.  The first grid barrier guarantees that every CTA has
            // flushed its peer stores before block zero enters the IPC
            // barrier.  Once block zero has completed that cross-rank
            // handshake it publishes a group-local ready epoch.  Other CTAs
            // wait on that single system word instead of paying a second grid
            // barrier, and no barrier is needed after the sum because the
            // next group writes disjoint scratch/output columns.
            __threadfence_system();
            a2a_grid_barrier<HipCooperativeLaunch>(grid_barrier);
            a2a_ipc_barrier(sync_buffer_ptrs, rank, world_size);
            int *const group_ready = grid_barrier + num_groups;
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                atomicExch_system(group_ready + group, epoch);
            }
            if (blockIdx.x != 0)
            {
                while (__hip_atomic_load(group_ready + group,
                                         __ATOMIC_ACQUIRE,
                                         __HIP_MEMORY_SCOPE_SYSTEM) != epoch)
                {
                }
            }
            a2a_sum_group(static_cast<const T *>(reduce_buffer_ptrs[rank]),
                          output,
                          ntokens_per_rank,
                          n_dim,
                          world_size,
                          group,
                          chunk_size_n);
            }
        }
    }

    if (group_pipeline)
    {
        // Keep the existing final lifetime handshake.  It protects scratch
        // reuse across operator invocations after the last group has been
        // reduced on every rank.
        __threadfence_system();
        a2a_grid_barrier<HipCooperativeLaunch>(grid_barrier);
        a2a_ipc_barrier(sync_buffer_ptrs, rank, world_size);
        return;
    }

    // All peer stores must reach system scope before any rank publishes the
    // first barrier flag.  Running this fence in every producer thread avoids
    // relying on one CTA lane to flush writes issued by other lanes.
    __threadfence_system();
    a2a_grid_barrier<HipCooperativeLaunch>(grid_barrier);
    a2a_ipc_barrier(sync_buffer_ptrs, rank, world_size);
    a2a_grid_barrier<HipCooperativeLaunch>(grid_barrier);

    T *local_buffer = static_cast<T *>(reduce_buffer_ptrs[rank]);
    const bool use_pack8 = (n_dim & 7) == 0;
    const int n_items = use_pack8 ? n_dim / 8 : n_dim;
    const int64_t total =
        static_cast<int64_t>(ntokens_per_rank) * n_items;
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_items);
        const int n_item = static_cast<int>(linear % n_items);
        const int n = use_pack8 ? n_item * 8 : n_item;
        T *destination = output + static_cast<int64_t>(token) * n_dim + n;
        if (use_pack8)
        {
            float values[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};
            for (int source_rank = 0; source_rank < world_size; ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                const ValuePack8<T> packed =
                    *reinterpret_cast<const ValuePack8<T> *>(
                        local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
                for (int lane = 0; lane < 8; ++lane)
                {
                    values[lane] += load_as_float<T>(packed.values[lane]);
                }
            }
            ValuePack8<T> packed_output;
#pragma unroll
            for (int lane = 0; lane < 8; ++lane)
            {
                packed_output.values[lane] =
                    store_from_float<T>(values[lane]);
            }
            *reinterpret_cast<ValuePack8<T> *>(destination) = packed_output;
        }
        else
        {
            float value = 0.0f;
            for (int source_rank = 0; source_rank < world_size; ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                value += load_as_float<T>(
                    local_buffer[static_cast<int64_t>(row) * n_dim + n]);
            }
            *destination = store_from_float<T>(value);
        }
    }

    // No peer may reuse its scratch allocation until every rank has finished
    // reading it.  The second IPC barrier retains the staged route's lifetime
    // contract; the final local barrier keeps non-zero blocks resident until
    // block zero has observed all peers.
    a2a_grid_barrier<HipCooperativeLaunch>(grid_barrier);
    a2a_ipc_barrier(sync_buffer_ptrs, rank, world_size);
    a2a_grid_barrier<HipCooperativeLaunch>(grid_barrier);
}

// A2A pipeline scatter stage.  This kernel deliberately has no grid-wide
// barrier: each CTA fences its peer writes, contributes to a per-group CTA
// counter in the source rank's signal allocation, and the final CTA publishes
// the group epoch.  A separate sum kernel consumes those epochs, allowing the
// two stages to run concurrently on different streams.
template <typename T, bool ApplyScale, bool UseTopkStage>
__global__ void a2a_signal_scatter_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch)
{
    const int ntokens_per_rank = ntokens / world_size;
    int *local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);

    for (int group = 0; group < num_groups; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        int this_chunk_size_n = n_dim - chunk_offset_n;
        if (this_chunk_size_n > chunk_size_n)
        {
            this_chunk_size_n = chunk_size_n;
        }
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const bool use_pack8 = (n_dim & 7) == 0 &&
                               (chunk_offset_n & 7) == 0 &&
                               (this_chunk_size_n & 7) == 0;
        const int n_items =
            use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
        const int64_t total = static_cast<int64_t>(ntokens) * n_items;
        for (int64_t linear =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int token = static_cast<int>(linear / n_items);
            const int n_item = static_cast<int>(linear % n_items);
            const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
            const int peer = token / ntokens_per_rank;
            const int local_token = token - peer * ntokens_per_rank;
            T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            const int dst_row = rank * ntokens_per_rank + local_token;
            T *destination =
                peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;

            if (UseTopkStage)
            {
                const float *source =
                    topk_stage + static_cast<int64_t>(token) * n_dim + n;
                if (use_pack8)
                {
                    const float4 first =
                        *reinterpret_cast<const float4 *>(source);
                    const float4 second =
                        *reinterpret_cast<const float4 *>(source + 4);
                    FloatPack8 values{{first.x, first.y, first.z, first.w,
                                       second.x, second.y, second.z, second.w}};
                    *reinterpret_cast<ValuePack8<T> *>(destination) =
                        cast_pack8<T>(values);
                }
                else
                {
                    *destination = store_from_float<T>(*source);
                }
            }
            else if (use_pack8)
            {
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                        gemm_output,
                        expert_weight,
                        token,
                        n,
                        n_dim,
                        topk));
            }
            else
            {
                *destination = store_from_float<T>(
                    reduce_topk_value<T, ApplyScale>(gemm_output,
                                                     expert_weight,
                                                     token,
                                                     n,
                                                     n_dim,
                                                     topk));
            }
        }

        // Every producer CTA flushes its own remote writes before entering
        // the CTA counter.  The system-scope atomic order makes the final
        // counter arrival a release point for the entire group.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const int old = atomicAdd_system(local_signals + group, 1);
            if (old + 1 == static_cast<int>(gridDim.x))
            {
                atomicExch_system(local_signals + signal_stride + group,
                                  epoch);
            }
        }
        __syncthreads();
    }
}

// One-group variant used by the hardware-wait route.  The AG completion
// dependency is supplied by hipStreamWaitValue32 on the host stream, so this
// kernel must never poll gemm_done_flags.  Keeping the ready publication in
// the kernel preserves the existing cross-rank IPC protocol.
template <typename T, bool ApplyScale, bool UseTopkStage>
__global__ void a2a_signal_scatter_one_group_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int group,
    int chunk_size_n,
    int signal_stride,
    int epoch)
{
    const int ntokens_per_rank = ntokens / world_size;
    const int chunk_offset_n = group * chunk_size_n;
    int this_chunk_size_n = n_dim - chunk_offset_n;
    if (this_chunk_size_n > chunk_size_n)
    {
        this_chunk_size_n = chunk_size_n;
    }
    if (this_chunk_size_n <= 0)
    {
        return;
    }

    int *local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    const bool use_pack8 = (n_dim & 7) == 0 &&
                           (chunk_offset_n & 7) == 0 &&
                           (this_chunk_size_n & 7) == 0;
    const int n_items =
        use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
    const int64_t total = static_cast<int64_t>(ntokens) * n_items;
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_items);
        const int n_item = static_cast<int>(linear % n_items);
        const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
        const int peer = token / ntokens_per_rank;
        const int local_token = token - peer * ntokens_per_rank;
        T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
        const int dst_row = rank * ntokens_per_rank + local_token;
        T *destination =
            peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;

        if (UseTopkStage)
        {
            const float *source =
                topk_stage + static_cast<int64_t>(token) * n_dim + n;
            if (use_pack8)
            {
                const float4 first =
                    *reinterpret_cast<const float4 *>(source);
                const float4 second =
                    *reinterpret_cast<const float4 *>(source + 4);
                FloatPack8 values{{first.x, first.y, first.z, first.w,
                                   second.x, second.y, second.z, second.w}};
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(values);
            }
            else
            {
                *destination = store_from_float<T>(*source);
            }
        }
        else if (use_pack8)
        {
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                    gemm_output,
                    expert_weight,
                    token,
                    n,
                    n_dim,
                    topk));
        }
        else
        {
            *destination = store_from_float<T>(reduce_topk_value<T, ApplyScale>(
                gemm_output,
                expert_weight,
                token,
                n,
                n_dim,
                topk));
        }
    }

    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd_system(local_signals + group, 1);
        if (old + 1 == static_cast<int>(gridDim.x))
        {
            atomicExch_system(local_signals + signal_stride + group, epoch);
        }
    }
    __syncthreads();
}

// Ragged-M variant of the one-group producer.  The equal-M producer derives
// the destination rank from token / (M / world); that mapping is invalid as
// soon as source ranks contribute different token counts.  The prefix offsets
// keep the gathered-GEMM token order intact while retaining a compact buffer
// layout on every destination rank: [producer_rank][destination_local_row].
template <typename T, bool ApplyScale, bool UseTopkStage>
__global__ void a2a_signal_scatter_one_group_ragged_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *rank_m_offsets,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int group,
    int chunk_size_n,
    int signal_stride,
    int epoch)
{
    const int chunk_offset_n = group * chunk_size_n;
    int this_chunk_size_n = n_dim - chunk_offset_n;
    if (this_chunk_size_n > chunk_size_n)
    {
        this_chunk_size_n = chunk_size_n;
    }
    if (this_chunk_size_n <= 0)
    {
        return;
    }

    int *local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    const bool use_pack8 = (n_dim & 7) == 0 &&
                           (chunk_offset_n & 7) == 0 &&
                           (this_chunk_size_n & 7) == 0;
    const int n_items = use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
    const int64_t total = static_cast<int64_t>(ntokens) * n_items;
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_items);
        const int n_item = static_cast<int>(linear % n_items);
        const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);

        int peer = 0;
        while (peer + 1 < world_size && token >= rank_m_offsets[peer + 1])
        {
            ++peer;
        }
        const int local_token = token - rank_m_offsets[peer];
        const int peer_rows = rank_m_offsets[peer + 1] - rank_m_offsets[peer];
        T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
        const int dst_row = rank * peer_rows + local_token;
        T *destination =
            peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;

        if (UseTopkStage)
        {
            const float *source =
                topk_stage + static_cast<int64_t>(token) * n_dim + n;
            if (use_pack8)
            {
                const float4 first = *reinterpret_cast<const float4 *>(source);
                const float4 second = *reinterpret_cast<const float4 *>(source + 4);
                FloatPack8 values{{first.x, first.y, first.z, first.w,
                                   second.x, second.y, second.z, second.w}};
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(values);
            }
            else
            {
                *destination = store_from_float<T>(*source);
            }
        }
        else if (use_pack8)
        {
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                    gemm_output, expert_weight, token, n, n_dim, topk));
        }
        else
        {
            *destination = store_from_float<T>(reduce_topk_value<T, ApplyScale>(
                gemm_output, expert_weight, token, n, n_dim, topk));
        }
    }

    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd_system(local_signals + group, 1);
        if (old + 1 == static_cast<int>(gridDim.x))
        {
            atomicExch_system(local_signals + signal_stride + group, epoch);
        }
    }
    __syncthreads();
}

// Persistent ragged producer for the stream-wait variant.  One launch loops
// over every N group and spins on the AG completion flag in-kernel instead of
// paying one host stream wait plus one launch per group.  The per-group
// signal publication matches the one-group kernel above.
template <typename T, bool ApplyScale, bool UseTopkStage>
__global__ void a2a_signal_scatter_ragged_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *rank_m_offsets,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch)
{
    int *local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    for (int group = 0; group < num_groups; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        int this_chunk_size_n = n_dim - chunk_offset_n;
        if (this_chunk_size_n > chunk_size_n)
        {
            this_chunk_size_n = chunk_size_n;
        }
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const bool use_pack8 = (n_dim & 7) == 0 &&
                               (chunk_offset_n & 7) == 0 &&
                               (this_chunk_size_n & 7) == 0;
        const int n_items =
            use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
        const int64_t total = static_cast<int64_t>(ntokens) * n_items;
        for (int64_t linear =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int token = static_cast<int>(linear / n_items);
            const int n_item = static_cast<int>(linear % n_items);
            const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);

            int peer = 0;
            while (peer + 1 < world_size && token >= rank_m_offsets[peer + 1])
            {
                ++peer;
            }
            const int local_token = token - rank_m_offsets[peer];
            const int peer_rows =
                rank_m_offsets[peer + 1] - rank_m_offsets[peer];
            T *peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            const int dst_row = rank * peer_rows + local_token;
            T *destination =
                peer_buffer + static_cast<int64_t>(dst_row) * n_dim + n;

            if (UseTopkStage)
            {
                const float *source =
                    topk_stage + static_cast<int64_t>(token) * n_dim + n;
                if (use_pack8)
                {
                    const float4 first =
                        *reinterpret_cast<const float4 *>(source);
                    const float4 second =
                        *reinterpret_cast<const float4 *>(source + 4);
                    FloatPack8 values{{first.x, first.y, first.z, first.w,
                                       second.x, second.y, second.z, second.w}};
                    *reinterpret_cast<ValuePack8<T> *>(destination) =
                        cast_pack8<T>(values);
                }
                else
                {
                    *destination = store_from_float<T>(*source);
                }
            }
            else if (use_pack8)
            {
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                        gemm_output, expert_weight, token, n, n_dim, topk));
            }
            else
            {
                *destination =
                    store_from_float<T>(reduce_topk_value<T, ApplyScale>(
                        gemm_output, expert_weight, token, n, n_dim, topk));
            }
        }

        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const int old = atomicAdd_system(local_signals + group, 1);
            if (old + 1 == static_cast<int>(gridDim.x))
            {
                atomicExch_system(local_signals + signal_stride + group, epoch);
            }
        }
        __syncthreads();
    }
}



// Consumer for the compact ragged destination layout above.  Every source
// rank contributes local_rows values to this rank, so source_rank * local_rows
// replaces the old source_rank * (M / world) address calculation.
template <typename T>
__global__ void a2a_signal_sum_rank_ragged_one_group_kernel(
    void **reduce_buffer_ptrs,
    T *output,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int group,
    int chunk_size_n)
{
    const int chunk_offset_n = group * chunk_size_n;
    int this_chunk_size_n = n_dim - chunk_offset_n;
    if (this_chunk_size_n > chunk_size_n)
    {
        this_chunk_size_n = chunk_size_n;
    }
    if (this_chunk_size_n <= 0)
    {
        return;
    }

    const bool use_pack8 = (n_dim & 7) == 0 &&
                           (chunk_offset_n & 7) == 0 &&
                           (this_chunk_size_n & 7) == 0;
    const int n_items = use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
    const int64_t total = static_cast<int64_t>(local_rows) * n_items;
    const T *local_buffer = static_cast<const T *>(reduce_buffer_ptrs[rank]);
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_items);
        const int n_item = static_cast<int>(linear % n_items);
        const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
        T *destination = output + static_cast<int64_t>(token) * n_dim + n;
        if (use_pack8)
        {
            float values[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};
            for (int source_rank = 0; source_rank < world_size; ++source_rank)
            {
                const int row = source_rank * local_rows + token;
                const ValuePack8<T> packed =
                    *reinterpret_cast<const ValuePack8<T> *>(
                        local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
                for (int lane = 0; lane < 8; ++lane)
                {
                    values[lane] += load_as_float<T>(packed.values[lane]);
                }
            }
            FloatPack8 values_pack;
#pragma unroll
            for (int lane = 0; lane < 8; ++lane)
            {
                values_pack.values[lane] = values[lane];
            }
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(values_pack);
        }
        else
        {
            float value = 0.0f;
            for (int source_rank = 0; source_rank < world_size; ++source_rank)
            {
                const int row = source_rank * local_rows + token;
                value += load_as_float<T>(
                    local_buffer[static_cast<int64_t>(row) * n_dim + n]);
            }
            *destination = store_from_float<T>(value);
        }
    }
}

// TP=8 ragged-M fast path for the common 512-column-subtile layout. Like the
// equal-M wave64 route, one wave owns one token and every lane transfers one
// aligned pack8. Ragged ownership is resolved only once by lane 0 and then
// broadcast to the wave, instead of having every pack8 perform the generic
// prefix-offset traversal.  For the direct-output topk=6 case, the epilogue
// also uses a compile-time-specialized reduction, matching the fixed-M
// wave64 producer's reduction contract.
template <typename T, bool ApplyScale, bool UseTopkStage, int DirectTopK, int WorldSize>
__global__ void a2a_signal_scatter_one_group_ragged_wave64_512x_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *rank_m_offsets,
    int ntokens,
    int n_dim,
    int rank,
    int topk,
    int group,
    int chunk_size_n,
    int this_chunk_size_n,
    int signal_stride,
    int epoch)
{
    static_assert(WorldSize >= 1 && WorldSize <= 8,
                  "ragged wave64/512x supports up to TP8");
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kSubtileSizeN = kWaveSize * 8;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    int token = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const int chunk_offset_n = group * chunk_size_n;
    const int num_subtiles = this_chunk_size_n / kSubtileSizeN;

    for (; token < ntokens; token += token_stride)
    {
        int wave_peer = 0;
        int wave_peer_start = 0;
        int wave_peer_rows = 0;
        if (lane == 0)
        {
#pragma unroll
            for (int candidate = 1; candidate < WorldSize; ++candidate)
            {
                wave_peer += token >= rank_m_offsets[candidate];
            }
            wave_peer_start = rank_m_offsets[wave_peer];
            wave_peer_rows = rank_m_offsets[wave_peer + 1] - wave_peer_start;
        }
        const int peer = __builtin_amdgcn_readfirstlane(wave_peer);
        const int peer_start =
            __builtin_amdgcn_readfirstlane(wave_peer_start);
        const int peer_rows =
            __builtin_amdgcn_readfirstlane(wave_peer_rows);
        const int local_token = token - peer_start;
        T *const peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
        T *const destination_row = peer_buffer +
            static_cast<int64_t>(rank * peer_rows + local_token) * n_dim;

        for (int subtile = 0; subtile < num_subtiles; ++subtile)
        {
            const int n = chunk_offset_n + subtile * kSubtileSizeN + lane * 8;
            T *const destination = destination_row + n;
            if constexpr (UseTopkStage)
            {
                const float *const source =
                    topk_stage + static_cast<int64_t>(token) * n_dim + n;
                const float4 first = *reinterpret_cast<const float4 *>(source);
                const float4 second =
                    *reinterpret_cast<const float4 *>(source + 4);
                FloatPack8 values{{first.x, first.y, first.z, first.w,
                                   second.x, second.y, second.z, second.w}};
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(values);
            }
            else
            {
                if constexpr (DirectTopK > 0)
                {
                    *reinterpret_cast<ValuePack8<T> *>(destination) =
                        cast_pack8<T>(
                            reduce_topk_static_pack8<T, ApplyScale, DirectTopK>(
                                gemm_output,
                                expert_weight,
                                token,
                                n,
                                n_dim));
                }
                else
                {
                    *reinterpret_cast<ValuePack8<T> *>(destination) =
                        cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                            gemm_output,
                            expert_weight,
                            token,
                            n,
                            n_dim,
                            topk));
                }
            }
        }
    }

    int *const local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd_system(local_signals + group, 1);
        if (old + 1 == static_cast<int>(gridDim.x))
        {
            atomicExch_system(local_signals + signal_stride + group, epoch);
        }
    }
    __syncthreads();
}

template <typename T,
          bool ApplyScale,
          bool UseTopkStage,
          int DirectTopK,
          int WorldSize>
__global__ void a2a_signal_scatter_ragged_wave64_512x_persistent_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *rank_m_offsets,
    const int *done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch)
{
    static_assert(WorldSize >= 1 && WorldSize <= 8,
                  "ragged wave64/512x supports up to TP8");
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kSubtileSizeN = kWaveSize * 8;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int token_start = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    int *const local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);

    for (int group = 0; group < num_groups; ++group)
    {
        wait_for_done_flag(done_flags, group);
        const int chunk_offset_n = group * chunk_size_n;
        int this_chunk_size_n = n_dim - chunk_offset_n;
        if (this_chunk_size_n > chunk_size_n)
        {
            this_chunk_size_n = chunk_size_n;
        }
        if (this_chunk_size_n <= 0)
        {
            continue;
        }
        const int num_subtiles = this_chunk_size_n / kSubtileSizeN;

        for (int token = token_start; token < ntokens; token += token_stride)
        {
            int wave_peer = 0;
            int wave_peer_start = 0;
            int wave_peer_rows = 0;
            if (lane == 0)
            {
#pragma unroll
                for (int candidate = 1; candidate < WorldSize; ++candidate)
                {
                    wave_peer += token >= rank_m_offsets[candidate];
                }
                wave_peer_start = rank_m_offsets[wave_peer];
                wave_peer_rows = rank_m_offsets[wave_peer + 1] - wave_peer_start;
            }
            const int peer = __builtin_amdgcn_readfirstlane(wave_peer);
            const int peer_start =
                __builtin_amdgcn_readfirstlane(wave_peer_start);
            const int peer_rows = __builtin_amdgcn_readfirstlane(wave_peer_rows);
            const int local_token = token - peer_start;
            T *const peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
            T *const destination_row =
                peer_buffer +
                static_cast<int64_t>(rank * peer_rows + local_token) * n_dim;

            for (int subtile = 0; subtile < num_subtiles; ++subtile)
            {
                const int n =
                    chunk_offset_n + subtile * kSubtileSizeN + lane * 8;
                T *const destination = destination_row + n;
                if constexpr (UseTopkStage)
                {
                    const float *const source =
                        topk_stage + static_cast<int64_t>(token) * n_dim + n;
                    const float4 first =
                        *reinterpret_cast<const float4 *>(source);
                    const float4 second =
                        *reinterpret_cast<const float4 *>(source + 4);
                    FloatPack8 values{{first.x, first.y, first.z, first.w,
                                       second.x, second.y, second.z, second.w}};
                    *reinterpret_cast<ValuePack8<T> *>(destination) =
                        cast_pack8<T>(values);
                }
                else if constexpr (DirectTopK > 0)
                {
                    *reinterpret_cast<ValuePack8<T> *>(destination) =
                        cast_pack8<T>(
                            reduce_topk_static_pack8<T, ApplyScale, DirectTopK>(
                                gemm_output, expert_weight, token, n, n_dim));
                }
                else
                {
                    *reinterpret_cast<ValuePack8<T> *>(destination) =
                        cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                            gemm_output, expert_weight, token, n, n_dim, topk));
                }
            }
        }

        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            const int old = atomicAdd_system(local_signals + group, 1);
            if (old + 1 == static_cast<int>(gridDim.x))
            {
                atomicExch_system(local_signals + signal_stride + group, epoch);
            }
        }
        __syncthreads();
    }
}

// Ragged counterpart of the TP=8 wave64 consumer.  The producer layout is
// [source_rank][destination_local_row], so only the per-rank row extent is
// dynamic; the eight-source reduction itself is fully static.
template <typename T, int WorldSize>
__global__ void a2a_signal_sum_rank_ragged_wave64_512x_kernel(
    void **reduce_buffer_ptrs,
    T *output,
    int local_rows,
    int n_dim,
    int rank,
    int group,
    int chunk_size_n,
    int this_chunk_size_n)
{
    static_assert(WorldSize >= 1 && WorldSize <= 8,
                  "ragged wave64/512x supports up to TP8");
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kSubtileSizeN = kWaveSize * 8;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    int token = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const int chunk_offset_n = group * chunk_size_n;
    const int num_subtiles = this_chunk_size_n / kSubtileSizeN;
    const T *const local_buffer =
        static_cast<const T *>(reduce_buffer_ptrs[rank]);

    for (; token < local_rows; token += token_stride)
    {
        const int64_t output_row_offset =
            static_cast<int64_t>(token) * n_dim;
        for (int subtile = 0; subtile < num_subtiles; ++subtile)
        {
            const int n = chunk_offset_n + subtile * kSubtileSizeN + lane * 8;
            FloatPack8 values{{0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
            for (int source_rank = 0; source_rank < WorldSize; ++source_rank)
            {
                const int row = source_rank * local_rows + token;
                const ValuePack8<T> packed =
                    *reinterpret_cast<const ValuePack8<T> *>(
                        local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
                for (int value = 0; value < 8; ++value)
                {
                    values.values[value] +=
                        load_as_float<T>(packed.values[value]);
                }
            }
            *reinterpret_cast<ValuePack8<T> *>(
                output + output_row_offset + n) = cast_pack8<T>(values);
        }
    }
}

// Shape-specialized producer for the 16384/chunks=32 fast route.  A gfx936
// wave contains 64 lanes, exactly the number of pack8 items in a 512-column
// group.  Mapping one wave to one token removes the generic loop's 64-bit
// divide/remainder operations while retaining the generic producer's token
// and peer traversal order.  Keeping the original reduce_topk_pack8 helper is
// intentional: fully unrolling topk and broadcasting every scale increased
// register/DS pressure and was slower on gfx936.
template <typename T, bool ApplyScale, bool UseTopkStage>
__global__ void a2a_signal_scatter_one_group_wave64_512_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    int ntokens,
    int n_dim,
    int rank,
    int group,
    int signal_stride,
    int epoch)
{
    constexpr int kWorldSize = 8;
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kChunkSizeN = kWaveSize * 8;

    const int ntokens_per_rank = ntokens / kWorldSize;
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    int token = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const int n = group * kChunkSizeN + lane * 8;
    const int dst_row_base = rank * ntokens_per_rank;

    for (; token < ntokens; token += token_stride)
    {
        const int peer = token / ntokens_per_rank;
        const int local_token = token - peer * ntokens_per_rank;
        T *const peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
        T *const destination = peer_buffer +
            static_cast<int64_t>(dst_row_base + local_token) * n_dim + n;
        if constexpr (UseTopkStage)
        {
            const float *const source =
                topk_stage + static_cast<int64_t>(token) * n_dim + n;
            const float4 first = *reinterpret_cast<const float4 *>(source);
            const float4 second =
                *reinterpret_cast<const float4 *>(source + 4);
            FloatPack8 values{{first.x, first.y, first.z, first.w,
                               second.x, second.y, second.z, second.w}};
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(values);
        }
        else
        {
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                    gemm_output,
                    expert_weight,
                    token,
                    n,
                    n_dim,
                    6));
        }
    }

    int *const local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd_system(local_signals + group, 1);
        if (old + 1 == static_cast<int>(gridDim.x))
        {
            atomicExch_system(local_signals + signal_stride + group, epoch);
        }
    }
    __syncthreads();
}

// Large logical N groups retain one AG done flag and one cross-rank ready
// epoch per group, but are traversed as 512-column wave64 subtiles.  One wave
// owns one token and reuses its token/peer address calculation across all
// subtiles.  This extends the division-free 512-column mapping to the
// 1024/2048/... groups selected for large N without expanding the signal ABI
// or multiplying the number of kernel launches.
template <typename T, bool ApplyScale, bool UseTopkStage>
__global__ void a2a_signal_scatter_one_group_wave64_512x_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    int ntokens,
    int n_dim,
    int rank,
    int group,
    int chunk_size_n,
    int this_chunk_size_n,
    int signal_stride,
    int epoch)
{
    constexpr int kWorldSize = 8;
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kSubtileSizeN = kWaveSize * 8;

    const int ntokens_per_rank = ntokens / kWorldSize;
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    int token = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const int chunk_offset_n = group * chunk_size_n;
    const int num_subtiles = this_chunk_size_n / kSubtileSizeN;
    const int dst_row_base = rank * ntokens_per_rank;

    for (; token < ntokens; token += token_stride)
    {
        const int peer = token / ntokens_per_rank;
        const int local_token = token - peer * ntokens_per_rank;
        T *const peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
        T *const destination_row = peer_buffer +
            static_cast<int64_t>(dst_row_base + local_token) * n_dim;

        for (int subtile = 0; subtile < num_subtiles; ++subtile)
        {
            const int n = chunk_offset_n + subtile * kSubtileSizeN + lane * 8;
            T *const destination = destination_row + n;
            if constexpr (UseTopkStage)
            {
                const float *const source =
                    topk_stage + static_cast<int64_t>(token) * n_dim + n;
                const float4 first = *reinterpret_cast<const float4 *>(source);
                const float4 second =
                    *reinterpret_cast<const float4 *>(source + 4);
                FloatPack8 values{{first.x, first.y, first.z, first.w,
                                   second.x, second.y, second.z, second.w}};
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(values);
            }
            else
            {
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                        gemm_output,
                        expert_weight,
                        token,
                        n,
                        n_dim,
                        6));
            }
        }
    }

    int *const local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd_system(local_signals + group, 1);
        if (old + 1 == static_cast<int>(gridDim.x))
        {
            atomicExch_system(local_signals + signal_stride + group, epoch);
        }
    }
    __syncthreads();
}

// For a 256-column group, each half-wave owns one token and every lane still
// moves one aligned pack8.  Keeping this as a separate kernel preserves the
// performance-sensitive 512-column kernel's original code generation.
template <typename T, bool ApplyScale, bool UseTopkStage>
__global__ void a2a_signal_scatter_one_group_wave64_256_kernel(
    const T *gemm_output,
    const T *expert_weight,
    const float *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    int ntokens,
    int n_dim,
    int rank,
    int group,
    int signal_stride,
    int epoch)
{
    constexpr int kWorldSize = 8;
    constexpr int kWaveSize = 64;
    constexpr int kHalfWaveSize = 32;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kTokensPerWave = 2;
    constexpr int kChunkSizeN = kHalfWaveSize * 8;

    const int ntokens_per_rank = ntokens / kWorldSize;
    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int token_in_wave = lane / kHalfWaveSize;
    const int pack_in_token = lane & (kHalfWaveSize - 1);
    int token = (blockIdx.x * kWavesPerBlock + wave) * kTokensPerWave +
                token_in_wave;
    const int token_stride =
        gridDim.x * kWavesPerBlock * kTokensPerWave;
    const int n = group * kChunkSizeN + pack_in_token * 8;
    const int dst_row_base = rank * ntokens_per_rank;

    for (; token < ntokens; token += token_stride)
    {
        const int peer = token / ntokens_per_rank;
        const int local_token = token - peer * ntokens_per_rank;
        T *const peer_buffer = static_cast<T *>(reduce_buffer_ptrs[peer]);
        T *const destination = peer_buffer +
            static_cast<int64_t>(dst_row_base + local_token) * n_dim + n;
        if constexpr (UseTopkStage)
        {
            const float *const source =
                topk_stage + static_cast<int64_t>(token) * n_dim + n;
            const float4 first = *reinterpret_cast<const float4 *>(source);
            const float4 second =
                *reinterpret_cast<const float4 *>(source + 4);
            FloatPack8 values{{first.x, first.y, first.z, first.w,
                               second.x, second.y, second.z, second.w}};
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(values);
        }
        else
        {
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(reduce_topk_pack8<T, ApplyScale>(
                    gemm_output,
                    expert_weight,
                    token,
                    n,
                    n_dim,
                    6));
        }
    }

    int *const local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int old = atomicAdd_system(local_signals + group, 1);
        if (old + 1 == static_cast<int>(gridDim.x))
        {
            atomicExch_system(local_signals + signal_stride + group, epoch);
        }
    }
    __syncthreads();
}

// A2A pipeline sum stage.  The signal allocation belongs to each source rank;
// every destination rank waits on all source-owned ready epochs before
// reducing one N group.  Groups are independent, so blocks may advance to the
// next group without a grid barrier while the scatter kernel is producing it.
template <typename T, bool WaitForSignal>
__global__ void a2a_signal_sum_rank_kernel(void **reduce_buffer_ptrs,
                                           void **signal_buffer_ptrs,
                                           T *output,
                                           int ntokens_per_rank,
                                           int n_dim,
                                           int rank,
                                           int world_size,
                                           int chunk_size_n,
                                           int group_begin,
                                           int group_end,
                                           int signal_stride,
                                           int epoch)
{
    int *const *signals = reinterpret_cast<int *const *>(signal_buffer_ptrs);
    T *local_buffer = static_cast<T *>(reduce_buffer_ptrs[rank]);
    int *local_signals = static_cast<int *>(signal_buffer_ptrs[rank]);
    (void)local_signals;

    for (int group = group_begin; group < group_end; ++group)
    {
        if constexpr (WaitForSignal)
        {
            if (threadIdx.x == 0)
            {
                for (int source_rank = 0; source_rank < world_size;
                     ++source_rank)
                {
                    int *source_signals = signals[source_rank];
                    while (__hip_atomic_load(
                               source_signals + signal_stride + group,
                               __ATOMIC_ACQUIRE,
                               __HIP_MEMORY_SCOPE_SYSTEM) != epoch)
                    {
                    }
                }
            }
            __syncthreads();
        }

        const int chunk_offset_n = group * chunk_size_n;
        int this_chunk_size_n = n_dim - chunk_offset_n;
        if (this_chunk_size_n > chunk_size_n)
        {
            this_chunk_size_n = chunk_size_n;
        }
        if (this_chunk_size_n <= 0)
        {
            continue;
        }
        const bool use_pack8 = (n_dim & 7) == 0 &&
                               (chunk_offset_n & 7) == 0 &&
                               (this_chunk_size_n & 7) == 0;
        const int n_items =
            use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
        const int64_t total =
            static_cast<int64_t>(ntokens_per_rank) * n_items;
        for (int64_t linear =
                 static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             linear < total;
             linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
        {
            const int token = static_cast<int>(linear / n_items);
            const int n_item = static_cast<int>(linear % n_items);
            const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
            T *destination =
                output + static_cast<int64_t>(token) * n_dim + n;
            if (use_pack8)
            {
                float values[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                                   0.0f, 0.0f, 0.0f, 0.0f};
                for (int source_rank = 0; source_rank < world_size;
                     ++source_rank)
                {
                    const int row = source_rank * ntokens_per_rank + token;
                    const ValuePack8<T> packed =
                        *reinterpret_cast<const ValuePack8<T> *>(
                            local_buffer + static_cast<int64_t>(row) * n_dim +
                            n);
#pragma unroll
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        values[lane] += load_as_float<T>(packed.values[lane]);
                    }
                }
                FloatPack8 values_pack;
#pragma unroll
                for (int lane = 0; lane < 8; ++lane)
                {
                    values_pack.values[lane] = values[lane];
                }
                *reinterpret_cast<ValuePack8<T> *>(destination) =
                    cast_pack8<T>(values_pack);
            }
            else
            {
                float value = 0.0f;
                for (int source_rank = 0; source_rank < world_size;
                     ++source_rank)
                {
                    const int row = source_rank * ntokens_per_rank + token;
                    value += load_as_float<T>(
                        local_buffer[static_cast<int64_t>(row) * n_dim + n]);
                }
                *destination = store_from_float<T>(value);
            }
        }
    }
}

// Matching wave64 consumer for one 512-column group.  Each wave owns one
// output token and each lane accumulates one aligned pack8 across all ranks.
// The fixed TP=8 loop and token mapping remove the generic kernel's per-item
// division/remainder and dynamic source-rank loop control.
template <typename T>
__global__ void a2a_signal_sum_rank_wave64_512_kernel(
    void **reduce_buffer_ptrs,
    T *output,
    int ntokens_per_rank,
    int n_dim,
    int rank,
    int group)
{
    constexpr int kWorldSize = 8;
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kChunkSizeN = kWaveSize * 8;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    int token = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const int n = group * kChunkSizeN + lane * 8;
    const T *const local_buffer =
        static_cast<const T *>(reduce_buffer_ptrs[rank]);

    for (; token < ntokens_per_rank; token += token_stride)
    {
        FloatPack8 values{{0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int source_rank = 0; source_rank < kWorldSize; ++source_rank)
        {
            const int row = source_rank * ntokens_per_rank + token;
            const ValuePack8<T> packed =
                *reinterpret_cast<const ValuePack8<T> *>(
                    local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
            for (int value = 0; value < 8; ++value)
            {
                values.values[value] +=
                    load_as_float<T>(packed.values[value]);
            }
        }
        *reinterpret_cast<ValuePack8<T> *>(
            output + static_cast<int64_t>(token) * n_dim + n) =
            cast_pack8<T>(values);
    }
}

// Matching large-group consumer.  The signal remains logical-group-major;
// each wave reduces all 512-column subtiles for one output token after that
// group's eight source-rank epochs have arrived.
template <typename T>
__global__ void a2a_signal_sum_rank_wave64_512x_kernel(
    void **reduce_buffer_ptrs,
    T *output,
    int ntokens_per_rank,
    int n_dim,
    int rank,
    int group,
    int chunk_size_n,
    int this_chunk_size_n)
{
    constexpr int kWorldSize = 8;
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kSubtileSizeN = kWaveSize * 8;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    int token = blockIdx.x * kWavesPerBlock + wave;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const int chunk_offset_n = group * chunk_size_n;
    const int num_subtiles = this_chunk_size_n / kSubtileSizeN;
    const T *const local_buffer =
        static_cast<const T *>(reduce_buffer_ptrs[rank]);

    for (; token < ntokens_per_rank; token += token_stride)
    {
        const int64_t output_row_offset =
            static_cast<int64_t>(token) * n_dim;
        for (int subtile = 0; subtile < num_subtiles; ++subtile)
        {
            const int n = chunk_offset_n + subtile * kSubtileSizeN + lane * 8;
            FloatPack8 values{{0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
            for (int source_rank = 0; source_rank < kWorldSize; ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                const ValuePack8<T> packed =
                    *reinterpret_cast<const ValuePack8<T> *>(
                        local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
                for (int value = 0; value < 8; ++value)
                {
                    values.values[value] +=
                        load_as_float<T>(packed.values[value]);
                }
            }
            *reinterpret_cast<ValuePack8<T> *>(
                output + output_row_offset + n) = cast_pack8<T>(values);
        }
    }
}

// Matching 256-column consumer: each half-wave reduces one output token.
template <typename T>
__global__ void a2a_signal_sum_rank_wave64_256_kernel(
    void **reduce_buffer_ptrs,
    T *output,
    int ntokens_per_rank,
    int n_dim,
    int rank,
    int group)
{
    constexpr int kWorldSize = 8;
    constexpr int kWaveSize = 64;
    constexpr int kHalfWaveSize = 32;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kTokensPerWave = 2;
    constexpr int kChunkSizeN = kHalfWaveSize * 8;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int token_in_wave = lane / kHalfWaveSize;
    const int pack_in_token = lane & (kHalfWaveSize - 1);
    int token = (blockIdx.x * kWavesPerBlock + wave) * kTokensPerWave +
                token_in_wave;
    const int token_stride =
        gridDim.x * kWavesPerBlock * kTokensPerWave;
    const int n = group * kChunkSizeN + pack_in_token * 8;
    const T *const local_buffer =
        static_cast<const T *>(reduce_buffer_ptrs[rank]);

    for (; token < ntokens_per_rank; token += token_stride)
    {
        FloatPack8 values{{0.0f, 0.0f, 0.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
        for (int source_rank = 0; source_rank < kWorldSize; ++source_rank)
        {
            const int row = source_rank * ntokens_per_rank + token;
            const ValuePack8<T> packed =
                *reinterpret_cast<const ValuePack8<T> *>(
                    local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
            for (int value = 0; value < 8; ++value)
            {
                values.values[value] +=
                    load_as_float<T>(packed.values[value]);
            }
        }
        *reinterpret_cast<ValuePack8<T> *>(
            output + static_cast<int64_t>(token) * n_dim + n) =
            cast_pack8<T>(values);
    }
}

template <typename T>
__device__ inline void a2a_sum_group(const T *local_buffer,
                                     T *output,
                                     int ntokens_per_rank,
                                     int n_dim,
                                     int world_size,
                                     int group,
                                     int chunk_size_n)
{
    const int chunk_offset_n = group * chunk_size_n;
    int this_chunk_size_n = n_dim - chunk_offset_n;
    if (this_chunk_size_n > chunk_size_n)
        this_chunk_size_n = chunk_size_n;
    if (this_chunk_size_n <= 0)
        return;
    const bool use_pack8 = (n_dim & 7) == 0 &&
                           (chunk_offset_n & 7) == 0 &&
                           (this_chunk_size_n & 7) == 0;
    const int n_items = use_pack8 ? this_chunk_size_n / 8 : this_chunk_size_n;
    const int64_t total = static_cast<int64_t>(ntokens_per_rank) * n_items;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_items);
        const int n_item = static_cast<int>(linear % n_items);
        const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
        T *destination = output + static_cast<int64_t>(token) * n_dim + n;
        if (use_pack8)
        {
            float values[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};
            for (int source_rank = 0; source_rank < world_size;
                 ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                const ValuePack8<T> packed =
                    *reinterpret_cast<const ValuePack8<T> *>(
                        local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
                for (int lane = 0; lane < 8; ++lane)
                    values[lane] += load_as_float<T>(packed.values[lane]);
            }
            FloatPack8 values_pack;
#pragma unroll
            for (int lane = 0; lane < 8; ++lane)
                values_pack.values[lane] = values[lane];
            *reinterpret_cast<ValuePack8<T> *>(destination) =
                cast_pack8<T>(values_pack);
        }
        else
        {
            float value = 0.0f;
            for (int source_rank = 0; source_rank < world_size;
                 ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                value += load_as_float<T>(
                    local_buffer[static_cast<int64_t>(row) * n_dim + n]);
            }
            *destination = store_from_float<T>(value);
        }
    }
}

template <typename T>
__global__ void sum_a2a_source_kernel(void **reduce_buffer_ptrs,
                                      T *output,
                                      int ntokens_per_rank,
                                      int n_dim,
                                      int rank,
                                      int world_size,
                                      int chunk_offset_n,
                                      int chunk_size_n)
{
    T *local_buffer = static_cast<T *>(reduce_buffer_ptrs[rank]);
    const bool use_pack8 = (n_dim & 7) == 0 &&
                           (chunk_offset_n & 7) == 0 &&
                           (chunk_size_n & 7) == 0;
    const int n_items = use_pack8 ? chunk_size_n / 8 : chunk_size_n;
    const int64_t total =
        static_cast<int64_t>(ntokens_per_rank) * n_items;
    for (int64_t linear =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int token = static_cast<int>(linear / n_items);
        const int n_item = static_cast<int>(linear % n_items);
        const int n = chunk_offset_n + (use_pack8 ? n_item * 8 : n_item);
        T *destination = output + static_cast<int64_t>(token) * n_dim + n;
        if (use_pack8)
        {
            float values[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f};
            for (int source_rank = 0; source_rank < world_size; ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                const ValuePack8<T> packed =
                    *reinterpret_cast<const ValuePack8<T> *>(
                        local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
                for (int lane = 0; lane < 8; ++lane)
                {
                    values[lane] += load_as_float<T>(packed.values[lane]);
                }
            }
            ValuePack8<T> packed_output;
#pragma unroll
            for (int lane = 0; lane < 8; ++lane)
            {
                packed_output.values[lane] =
                    store_from_float<T>(values[lane]);
            }
            *reinterpret_cast<ValuePack8<T> *>(destination) = packed_output;
        }
        else
        {
            float value = 0.0f;
            for (int source_rank = 0; source_rank < world_size; ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                value += load_as_float<T>(
                    local_buffer[static_cast<int64_t>(row) * n_dim + n]);
            }
            *destination = store_from_float<T>(value);
        }
    }
}

// Division-free TP<=8 consumer for the aligned large-N contract.  Each wave
// owns one output token and one lane writes an aligned pack8 for each
// 512-column subtile.  WorldSize is a template parameter so TP2/TP4 keep the
// same vectorized layout while preserving the TP8 unrolled source loop.
template <typename T, int kWorldSize>
__global__ void sum_a2a_source_wave64_512x_kernel(void **reduce_buffer_ptrs,
                                                  T *output,
                                                  int ntokens_per_rank,
                                                  int n_dim,
                                                  int rank)
{
    static_assert(kWorldSize >= 1 && kWorldSize <= 8,
                  "wave64/512x A2A sum supports up to TP8");
    constexpr int kWaveSize = 64;
    constexpr int kWavesPerBlock = kReduceThreads / kWaveSize;
    constexpr int kSubtileSizeN = kWaveSize * 8;

    const int lane = threadIdx.x & (kWaveSize - 1);
    const int wave = threadIdx.x / kWaveSize;
    const int token_stride = gridDim.x * kWavesPerBlock;
    const T *const local_buffer =
        static_cast<const T *>(reduce_buffer_ptrs[rank]);
    const int num_subtiles = n_dim / kSubtileSizeN;

    for (int token = blockIdx.x * kWavesPerBlock + wave;
         token < ntokens_per_rank;
         token += token_stride)
    {
        for (int subtile = 0; subtile < num_subtiles; ++subtile)
        {
            const int n = subtile * kSubtileSizeN + lane * 8;
            FloatPack8 values{{0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f}};
#pragma unroll
            for (int source_rank = 0; source_rank < kWorldSize;
                 ++source_rank)
            {
                const int row = source_rank * ntokens_per_rank + token;
                const ValuePack8<T> packed =
                    *reinterpret_cast<const ValuePack8<T> *>(
                        local_buffer + static_cast<int64_t>(row) * n_dim + n);
#pragma unroll
                for (int value = 0; value < 8; ++value)
                {
                    values.values[value] +=
                        load_as_float<T>(packed.values[value]);
                }
            }
            *reinterpret_cast<ValuePack8<T> *>(
                output + static_cast<int64_t>(token) * n_dim + n) =
                cast_pack8<T>(values);
        }
    }
}



// Device-signal Ring variant.  Unlike ring_step_source_kernel, this kernel
// keeps all Ring steps in one persistent launch.  Each rank publishes a
// per-(group, segment) ready epoch into the next rank's uncached IPC signal
// buffer after its blocks have finished writing the segment.  The full-N
// scratch stride used by this route gives different N groups disjoint storage
// so the host does not need a barrier between group launches.

int capped_blocks(int64_t total)
{
    int blocks = static_cast<int>((total + kReduceThreads - 1) /
                                  kReduceThreads);
    if (blocks < 1)
    {
        blocks = 1;
    }
    if (blocks > kReduceBlocks)
    {
        blocks = kReduceBlocks;
    }
    return blocks;
}


int capped_work_blocks(int64_t total)
{
    int blocks = static_cast<int>((total + kReduceThreads - 1) /
                                  kReduceThreads);
    if (blocks < 1)
    {
        blocks = 1;
    }
    if (blocks > kWorkBlocks)
    {
        blocks = kWorkBlocks;
    }
    return blocks;
}




template <typename T, bool ApplyScale>
void launch_topk_scatter_a2a_groups(const void *gemm_output,
                                    const void *expert_weight,
                                    void **reduce_buffer_ptrs,
                                    const int *done_flags,
                                    int ntokens,
                                    int n_dim,
                                    int rank,
                                    int world_size,
                                    int topk,
                                    int chunk_size_n,
                                    int first_group,
                                    int group_count,
                                    int block_budget,
                                    hipStream_t stream)
{
    const int64_t largest_group =
        static_cast<int64_t>(ntokens) * chunk_size_n;
    int scatter_blocks = capped_blocks(largest_group);
    // The ordinary route intentionally starts one persistent multi-group
    // scatter before the AG GEMM has completed. Cap that grid when the caller
    // supplies a measured overlap budget; the uncapped path remains available
    // to the other launchers and for explicit ablations.
    if (block_budget > 0)
    {
        // A positive range budget is an explicit resource contract.  Keep
        // early overlap launches small, but allow a post-GEMM range to use
        // more than the generic 32-block cap when requested by the host.
        scatter_blocks = std::max(
            1,
            std::min(capped_work_blocks(largest_group), block_budget));
    }
    else
    {
        const char *budget_value =
            detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_SCATTER_BLOCK_BUDGET");
        if (budget_value != nullptr && *budget_value != '\0')
        {
            const int requested = std::atoi(budget_value);
            if (requested > 0)
            {
                scatter_blocks =
                    std::max(1, std::min(scatter_blocks, requested));
            }
        }
    }
    const char *wave64_512x_value =
        detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_ORDINARY_WAVE64_512X");
    const bool wave64_512x_requested =
        wave64_512x_value != nullptr && wave64_512x_value[0] != '\0' &&
        wave64_512x_value[0] != '0' && wave64_512x_value[0] != 'f' &&
        wave64_512x_value[0] != 'F' && wave64_512x_value[0] != 'n' &&
        wave64_512x_value[0] != 'N';
    const bool use_wave64_512x =
        world_size == 8 && n_dim % 512 == 0 && chunk_size_n % 512 == 0 &&
        wave64_512x_requested;
    if (use_wave64_512x)
    {
        hipLaunchKernelGGL(
            (topk_scatter_a2a_groups_wave64_512x_kernel<T, ApplyScale>),
            dim3(scatter_blocks),
            dim3(kReduceThreads),
            0,
            stream,
            static_cast<const T *>(gemm_output),
            static_cast<const T *>(expert_weight),
            reduce_buffer_ptrs,
            done_flags,
            ntokens,
            n_dim,
            rank,
            topk,
            chunk_size_n,
            first_group,
            group_count);
    }
    else
    {
        hipLaunchKernelGGL((topk_scatter_a2a_groups_kernel<T, ApplyScale>),
                           dim3(scatter_blocks),
                           dim3(kReduceThreads),
                           0,
                           stream,
                           static_cast<const T *>(gemm_output),
                           static_cast<const T *>(expert_weight),
                           reduce_buffer_ptrs,
                           done_flags,
                           ntokens,
                           n_dim,
                           rank,
                           world_size,
                           topk,
                           chunk_size_n,
                           first_group,
                           group_count);
    }
}

template <typename T, bool ApplyScale>
void launch_topk_scatter_a2a_paced_groups(const void *gemm_output,
                                          const void *expert_weight,
                                          void **reduce_buffer_ptrs,
                                          const int *done_flags,
                                          int ntokens,
                                          int n_dim,
                                          int rank,
                                          int world_size,
                                          int topk,
                                          int chunk_size_n,
                                          int num_groups,
                                          int first_group,
                                          int group_count,
                                          int block_budget,
                                          int check_iterations,
                                          int sleep_steps,
                                          hipStream_t stream)
{
    const int64_t largest_group =
        static_cast<int64_t>(ntokens) * chunk_size_n;
    const int scatter_blocks = std::max(
        1, std::min(capped_work_blocks(largest_group), block_budget));
    hipLaunchKernelGGL((topk_scatter_a2a_paced_groups_kernel<T, ApplyScale>),
                       dim3(scatter_blocks),
                       dim3(kReduceThreads),
                       0,
                       stream,
                       static_cast<const T *>(gemm_output),
                       static_cast<const T *>(expert_weight),
                       reduce_buffer_ptrs,
                       done_flags,
                       ntokens,
                       n_dim,
                       rank,
                       world_size,
                       topk,
                       chunk_size_n,
                       num_groups,
                       first_group,
                       group_count,
                       check_iterations,
                       sleep_steps);
}

template <typename T, bool ApplyScale>
void launch_topk_scatter_a2a_chunk(const void *gemm_output,
                                   const void *expert_weight,
                                   void **reduce_buffer_ptrs,
                                   const int *done_flags,
                                   int ntokens,
                                   int n_dim,
                                   int rank,
                                   int world_size,
                                   int topk,
                                   int group,
                                   int chunk_offset_n,
                                   int chunk_size_n,
                                   hipStream_t stream)
{
    const int64_t total = static_cast<int64_t>(ntokens) * chunk_size_n;
    hipLaunchKernelGGL((topk_scatter_a2a_chunk_kernel<T, ApplyScale>),
                       dim3(capped_blocks(total)),
                       dim3(kReduceThreads),
                       0,
                       stream,
                       static_cast<const T *>(gemm_output),
                       static_cast<const T *>(expert_weight),
                       reduce_buffer_ptrs,
                       done_flags,
                       ntokens,
                       n_dim,
                       rank,
                       world_size,
                       topk,
                       group,
                       chunk_offset_n,
                       chunk_size_n);
}

template <typename T,
          bool ApplyScale,
          bool UseTopkStage,
          bool HipCooperativeLaunch,
          bool Ragged>
void launch_a2a_cooperative_fused_impl(const void *gemm_output,
                                       const void *expert_weight,
                                       const void *topk_stage,
                                       void **reduce_buffer_ptrs,
                                       int **sync_buffer_ptrs,
                                       int *grid_barrier,
                                       int *gemm_done_flags,
                                       void *output,
                                       const int *rank_m_offsets,
                                       int local_rows,
                                       int ntokens,
                                       int n_dim,
                                       int rank,
                                       int world_size,
                                       int topk,
                                       int chunk_size_n,
                                       int num_groups,
                                       int epoch,
                                       bool group_pipeline,
                                       hipStream_t stream)
{
    const int largest_chunk_n =
        chunk_size_n < n_dim ? chunk_size_n : n_dim;
    const int64_t largest_group =
        static_cast<int64_t>(ntokens) * largest_chunk_n;
    int blocks = capped_blocks(largest_group);
    if (group_pipeline)
    {
        // The standard [K,N] grouped-GEMM route is much more sensitive to a
        // resident peer-scatter grid than the old packed-weight route.  Eight
        // cooperative blocks preserve group overlap without turning the
        // scatter into a competing full-device memory workload.  This
        // workload-based default applies to every group-pipeline shape; the
        // optional override is retained for hardware/shape ablations.
        int block_budget = 8;
        const char *value =
            detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_COOP_BLOCK_BUDGET");
        if (value != nullptr && *value != '\0')
        {
            const int requested = std::atoi(value);
            if (requested > 0)
            {
                block_budget = requested;
            }
        }
        blocks = std::max(1, std::min(blocks, block_budget));
    }
    auto kernel = a2a_cooperative_fused_kernel<T,
                                               ApplyScale,
                                               UseTopkStage,
                                               HipCooperativeLaunch,
                                               Ragged>;

    const T *typed_gemm_output = static_cast<const T *>(gemm_output);
    const T *typed_expert_weight = static_cast<const T *>(expert_weight);
    const float *typed_topk_stage = static_cast<const float *>(topk_stage);
    T *typed_output = static_cast<T *>(output);
    void *args[] = {&typed_gemm_output,
                    &typed_expert_weight,
                    &typed_topk_stage,
                    &reduce_buffer_ptrs,
                    &sync_buffer_ptrs,
                    &grid_barrier,
                    &gemm_done_flags,
                    &typed_output,
                    &rank_m_offsets,
                    &local_rows,
                    &ntokens,
                    &n_dim,
                    &rank,
                    &world_size,
                    &topk,
                    &chunk_size_n,
                    &num_groups,
                    &epoch,
                    &group_pipeline};

    if constexpr (HipCooperativeLaunch)
    {
        int device = 0;
        CUDA_CHECK(hipGetDevice(&device));
        struct CooperativeCapacity
        {
            int device = -1;
            int block_limit = 0;
        };
        // This helper is instantiated separately for each dtype/epilogue
        // variant.  Cache its occupancy on the calling host thread so the
        // steady-state operator path does not query kernel attributes on every
        // invocation, while still handling a later device switch correctly.
        static thread_local CooperativeCapacity capacity;
        if (capacity.device != device)
        {
            int cooperative_supported = 0;
            int active_blocks_per_cu = 0;
            hipDeviceProp_t properties{};
            CUDA_CHECK(hipDeviceGetAttribute(&cooperative_supported,
                                             hipDeviceAttributeCooperativeLaunch,
                                             device));
            FTRAIN_HYGON_CHECK(cooperative_supported != 0)
                << "HIP device does not support cooperative launch";
            CUDA_CHECK(hipGetDeviceProperties(&properties, device));
            CUDA_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_cu,
                reinterpret_cast<const void *>(kernel),
                kReduceThreads,
                0));
            capacity.device = device;
            capacity.block_limit =
                properties.multiProcessorCount * active_blocks_per_cu;
        }
        const int cooperative_block_limit = capacity.block_limit;
        FTRAIN_HYGON_CHECK(cooperative_block_limit > 0)
            << "fused A2A cooperative kernel has zero resident capacity";
        if (blocks > cooperative_block_limit)
        {
            blocks = cooperative_block_limit;
        }

        CUDA_CHECK(hipLaunchCooperativeKernel(
            reinterpret_cast<const void *>(kernel),
            dim3(blocks),
            dim3(kReduceThreads),
            args,
            0,
            stream));
    }
    else
    {
        CUDA_CHECK(hipLaunchKernel(reinterpret_cast<const void *>(kernel),
                                   dim3(blocks),
                                   dim3(kReduceThreads),
                                   args,
                                   0,
                                   stream));
    }
}

template <typename T, bool ApplyScale, bool UseTopkStage, bool Ragged>
void launch_a2a_cooperative_fused(const void *gemm_output,
                                  const void *expert_weight,
                                  const void *topk_stage,
                                  void **reduce_buffer_ptrs,
                                  int **sync_buffer_ptrs,
                                  int *grid_barrier,
                                  int *gemm_done_flags,
                                  void *output,
                                  const int *rank_m_offsets,
                                  int local_rows,
                                  int ntokens,
                                  int n_dim,
                                  int rank,
                                  int world_size,
                                  int topk,
                                  int chunk_size_n,
                                  int num_groups,
                                  int epoch,
                                  bool group_pipeline,
                                  bool hip_cooperative_launch,
                                  hipStream_t stream)
{
    if (hip_cooperative_launch)
    {
        launch_a2a_cooperative_fused_impl<T,
                                          ApplyScale,
                                          UseTopkStage,
                                          true,
                                          Ragged>(gemm_output,
                                                expert_weight,
                                                topk_stage,
                                                reduce_buffer_ptrs,
                                                sync_buffer_ptrs,
                                                grid_barrier,
                                                gemm_done_flags,
                                                output,
                                                rank_m_offsets,
                                                local_rows,
                                                ntokens,
                                                n_dim,
                                                rank,
                                                world_size,
                                                topk,
                                                chunk_size_n,
                                                num_groups,
                                                epoch,
                                                group_pipeline,
                                                stream);
        return;
    }
    launch_a2a_cooperative_fused_impl<T,
                                      ApplyScale,
                                      UseTopkStage,
                                      false,
                                      Ragged>(gemm_output,
                                             expert_weight,
                                             topk_stage,
                                             reduce_buffer_ptrs,
                                             sync_buffer_ptrs,
                                             grid_barrier,
                                             gemm_done_flags,
                                             output,
                                             rank_m_offsets,
                                             local_rows,
                                             ntokens,
                                             n_dim,
                                             rank,
                                             world_size,
                                             topk,
                                             chunk_size_n,
                                             num_groups,
                                             epoch,
                                             group_pipeline,
                                             stream);
}

template <typename T, bool ApplyScale, bool UseTopkStage>
void launch_a2a_cooperative_fused_dispatch(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    int **sync_buffer_ptrs,
    int *grid_barrier,
    int *gemm_done_flags,
    void *output,
    const int *rank_m_offsets,
    int local_rows,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int epoch,
    bool group_pipeline,
    bool hip_cooperative_launch,
    hipStream_t stream)
{
    if (rank_m_offsets != nullptr)
    {
        launch_a2a_cooperative_fused<T, ApplyScale, UseTopkStage, true>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            sync_buffer_ptrs,
            grid_barrier,
            gemm_done_flags,
            output,
            rank_m_offsets,
            local_rows,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            epoch,
            group_pipeline,
            hip_cooperative_launch,
            stream);
        return;
    }
    launch_a2a_cooperative_fused<T, ApplyScale, UseTopkStage, false>(
        gemm_output,
        expert_weight,
        topk_stage,
        reduce_buffer_ptrs,
        sync_buffer_ptrs,
        grid_barrier,
        gemm_done_flags,
        output,
        nullptr,
        0,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        epoch,
        group_pipeline,
        hip_cooperative_launch,
        stream);
}

template <typename T, bool ApplyScale, bool UseTopkStage>
void launch_a2a_signal_pipeline_impl(const void *gemm_output,
                                     const void *expert_weight,
                                     const void *topk_stage,
                                     void **reduce_buffer_ptrs,
                                     void **signal_buffer_ptrs,
                                     const int *gemm_done_flags,
                                     void *output,
                                     int ntokens,
                                     int n_dim,
                                     int rank,
                                     int world_size,
                                     int topk,
                                     int chunk_size_n,
                                     int num_groups,
                                     int signal_stride,
                                     int epoch,
                                     int block_budget,
                                     hipStream_t scatter_stream,
                                     hipStream_t sum_stream)
{
    const int budget =
        max(1, min(kReduceBlocks, block_budget));
    const int largest_chunk_n = min(n_dim, chunk_size_n);
    const int64_t scatter_total =
        static_cast<int64_t>(ntokens) * largest_chunk_n;
    const int64_t sum_total =
        static_cast<int64_t>(ntokens / world_size) * largest_chunk_n;
    int scatter_blocks = min(budget, capped_blocks(scatter_total));
    int sum_blocks = min(budget, capped_blocks(sum_total));
    if (scatter_blocks < 1)
    {
        scatter_blocks = 1;
    }
    if (sum_blocks < 1)
    {
        sum_blocks = 1;
    }

    hipLaunchKernelGGL(
        (a2a_signal_scatter_kernel<T, ApplyScale, UseTopkStage>),
        dim3(scatter_blocks),
        dim3(kReduceThreads),
        0,
        scatter_stream,
        static_cast<const T *>(gemm_output),
        static_cast<const T *>(expert_weight),
        static_cast<const float *>(topk_stage),
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        gemm_done_flags,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch);

    hipLaunchKernelGGL((a2a_signal_sum_rank_kernel<T, true>),
                       dim3(sum_blocks),
                       dim3(kReduceThreads),
                       0,
                       sum_stream,
                       reduce_buffer_ptrs,
                       signal_buffer_ptrs,
                       static_cast<T *>(output),
                       ntokens / world_size,
                       n_dim,
                       rank,
                       world_size,
                       chunk_size_n,
                       0,
                       num_groups,
                       signal_stride,
                       epoch);
}

template <typename T, bool ApplyScale, bool UseTopkStage>
void launch_a2a_signal_pipeline_split_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int producer_block_budget,
    int consumer_block_budget,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    const int producer_budget =
        max(1, min(kReduceBlocks, producer_block_budget));
    const int consumer_budget =
        max(1, min(kReduceBlocks, consumer_block_budget));
    const int largest_chunk_n = min(n_dim, chunk_size_n);
    const int64_t scatter_total =
        static_cast<int64_t>(ntokens) * largest_chunk_n;
    const int64_t sum_total =
        static_cast<int64_t>(ntokens / world_size) * largest_chunk_n;
    const int scatter_blocks =
        max(1, min(producer_budget, capped_blocks(scatter_total)));
    const int sum_blocks =
        max(1, min(consumer_budget, capped_blocks(sum_total)));

    hipLaunchKernelGGL(
        (a2a_signal_scatter_kernel<T, ApplyScale, UseTopkStage>),
        dim3(scatter_blocks),
        dim3(kReduceThreads),
        0,
        scatter_stream,
        static_cast<const T *>(gemm_output),
        static_cast<const T *>(expert_weight),
        static_cast<const float *>(topk_stage),
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        gemm_done_flags,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch);

    hipLaunchKernelGGL((a2a_signal_sum_rank_kernel<T, true>),
                       dim3(sum_blocks),
                       dim3(kReduceThreads),
                       0,
                       sum_stream,
                       reduce_buffer_ptrs,
                       signal_buffer_ptrs,
                       static_cast<T *>(output),
                       ntokens / world_size,
                       n_dim,
                       rank,
                       world_size,
                       chunk_size_n,
                       0,
                       num_groups,
                       signal_stride,
                       epoch);
}

template <typename T, bool ApplyScale>
void launch_a2a_signal_pipeline_split_dtype(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int producer_block_budget,
    int consumer_block_budget,
    bool use_topk_stage,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    if (use_topk_stage)
    {
        launch_a2a_signal_pipeline_split_impl<T, ApplyScale, true>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            producer_block_budget,
            consumer_block_budget,
            scatter_stream,
            sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_split_impl<T, ApplyScale, false>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            producer_block_budget,
            consumer_block_budget,
            scatter_stream,
            sum_stream);
    }
}

template <typename T, bool ApplyScale>
void launch_a2a_signal_pipeline_dtype(const void *gemm_output,
                                      const void *expert_weight,
                                      const void *topk_stage,
                                      void **reduce_buffer_ptrs,
                                      void **signal_buffer_ptrs,
                                      const int *gemm_done_flags,
                                      void *output,
                                      int ntokens,
                                      int n_dim,
                                      int rank,
                                      int world_size,
                                      int topk,
                                      int chunk_size_n,
                                      int num_groups,
                                      int signal_stride,
                                      int epoch,
                                      int block_budget,
                                      bool use_topk_stage,
                                      hipStream_t scatter_stream,
                                      hipStream_t sum_stream)
{
    if (use_topk_stage)
    {
        launch_a2a_signal_pipeline_impl<T, ApplyScale, true>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            block_budget,
            scatter_stream,
            sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_impl<T, ApplyScale, false>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            block_budget,
            scatter_stream,
            sum_stream);
    }
}

// Variant used by the stream-wait pipeline.  Unlike the persistent sum
// kernel above, every group is a short ordinary launch preceded by stream
// wait-value commands.  Waits are handled by the scheduler rather than by a
// resident CTA, which leaves the rank-tile GEMM eligible to run while a group
// is still in flight.
template <typename T, bool ApplyScale, bool UseTopkStage>
void launch_a2a_signal_pipeline_stream_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int block_budget,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    const int budget = max(1, min(kReduceBlocks, block_budget));
    const int largest_chunk_n = min(n_dim, chunk_size_n);
    const int64_t scatter_total =
        static_cast<int64_t>(ntokens) * largest_chunk_n;
    const int64_t sum_total =
        static_cast<int64_t>(ntokens / world_size) * largest_chunk_n;
    int scatter_blocks = min(budget, capped_blocks(scatter_total));
    int sum_blocks = min(budget, capped_blocks(sum_total));
    if (scatter_blocks < 1)
        scatter_blocks = 1;
    if (sum_blocks < 1)
        sum_blocks = 1;

    hipLaunchKernelGGL(
        (a2a_signal_scatter_kernel<T, ApplyScale, UseTopkStage>),
        dim3(scatter_blocks),
        dim3(kReduceThreads),
        0,
        scatter_stream,
        static_cast<const T *>(gemm_output),
        static_cast<const T *>(expert_weight),
        static_cast<const float *>(topk_stage),
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        gemm_done_flags,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch);

    // A wait-value command does not consume a CTA while it is pending.  All
    // source-rank epochs are required before launching this destination's
    // local sum for a group.  The signal pointers are IPC-visible and are
    // intentionally supplied in host form for the stream API.
    for (int group = 0; group < num_groups; ++group)
    {
        for (int source_rank = 0; source_rank < world_size; ++source_rank)
        {
            void *signal = const_cast<void *>(signal_buffer_ptrs_host[source_rank]);
            CUDA_CHECK(hipStreamWaitValue32(
                sum_stream,
                static_cast<char *>(signal) +
                    static_cast<size_t>(signal_stride + group) * sizeof(int),
                static_cast<uint32_t>(epoch),
                hipStreamWaitValueGte,
                0xffffffffu));
        }
        const int chunk_offset_n = group * chunk_size_n;
        const int this_chunk_size_n =
            min(chunk_size_n, n_dim - chunk_offset_n);
        if (this_chunk_size_n <= 0)
            continue;
        hipLaunchKernelGGL((a2a_signal_sum_rank_kernel<T, false>),
                           dim3(sum_blocks),
                           dim3(kReduceThreads),
                           0,
                           sum_stream,
                           reduce_buffer_ptrs,
                           signal_buffer_ptrs,
                           static_cast<T *>(output),
                           ntokens / world_size,
                           n_dim,
                           rank,
                           world_size,
                           chunk_size_n,
                           group,
                           group + 1,
                           signal_stride,
                           epoch);
    }
}

template <typename T, bool ApplyScale, bool UseTopkStage>
void launch_a2a_signal_pipeline_hardware_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    int early_scatter_block_budget,
    int early_sum_block_budget,
    int full_budget_start_group,
    bool wave64_512_scatter,
    bool wave64_512_sum,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    const bool use_wave64_scatter =
        wave64_512_scatter && world_size == 8 && topk == 6;
    const bool use_wave64_sum = wave64_512_sum && world_size == 8;
    const int scatter_budget =
        max(1, min(kReduceBlocks, scatter_block_budget));
    const int sum_budget = max(1, min(kReduceBlocks, sum_block_budget));
    const int early_scatter_budget =
        max(1, min(kReduceBlocks, early_scatter_block_budget));
    const int early_sum_budget =
        max(1, min(kReduceBlocks, early_sum_block_budget));
    const int full_budget_group = max(0, min(num_groups,
                                             full_budget_start_group));
    for (int group = 0; group < num_groups; ++group)
    {
        const int chunk_offset_n = group * chunk_size_n;
        const int this_chunk_size_n =
            min(chunk_size_n, n_dim - chunk_offset_n);
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const int64_t scatter_total =
            static_cast<int64_t>(ntokens) * this_chunk_size_n;
        const int64_t sum_total =
            static_cast<int64_t>(ntokens / world_size) * this_chunk_size_n;
        const int group_scatter_budget =
            group < full_budget_group ? early_scatter_budget : scatter_budget;
        const int group_sum_budget =
            group < full_budget_group ? early_sum_budget : sum_budget;
        const int scatter_blocks =
            max(1, min(group_scatter_budget, capped_blocks(scatter_total)));
        const int sum_blocks =
            max(1, min(group_sum_budget, capped_blocks(sum_total)));
        // Unlike wait_for_done_flag(), this scheduler wait does not occupy a
        // resident CTA while the AG rank-tile object is still computing.
        CUDA_CHECK(hipStreamWaitValue32(
            scatter_stream,
            const_cast<int *>(gemm_done_flags + group),
            1u,
            hipStreamWaitValueGte,
            0xffffffffu));
        const bool use_wave64_512x_scatter =
            use_wave64_scatter && chunk_size_n > 512 &&
            this_chunk_size_n % 512 == 0;
        const bool use_wave64_512x_sum =
            use_wave64_sum && chunk_size_n > 512 &&
            this_chunk_size_n % 512 == 0;
        if (use_wave64_scatter && chunk_size_n == 512 &&
            this_chunk_size_n == 512)
        {
            hipLaunchKernelGGL(
                (a2a_signal_scatter_one_group_wave64_512_kernel<
                    T,
                    ApplyScale,
                    UseTopkStage>),
                dim3(scatter_blocks),
                dim3(kReduceThreads),
                0,
                scatter_stream,
                static_cast<const T *>(gemm_output),
                static_cast<const T *>(expert_weight),
                static_cast<const float *>(topk_stage),
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                ntokens,
                n_dim,
                rank,
                group,
                signal_stride,
                epoch);
        }
        else if (use_wave64_scatter && chunk_size_n == 256 &&
                 this_chunk_size_n == 256)
        {
            hipLaunchKernelGGL(
                (a2a_signal_scatter_one_group_wave64_256_kernel<
                    T,
                    ApplyScale,
                    UseTopkStage>),
                dim3(scatter_blocks),
                dim3(kReduceThreads),
                0,
                scatter_stream,
                static_cast<const T *>(gemm_output),
                static_cast<const T *>(expert_weight),
                static_cast<const float *>(topk_stage),
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                ntokens,
                n_dim,
                rank,
                group,
                signal_stride,
                epoch);
        }
        else if (use_wave64_512x_scatter)
        {
            hipLaunchKernelGGL(
                (a2a_signal_scatter_one_group_wave64_512x_kernel<
                    T,
                    ApplyScale,
                    UseTopkStage>),
                dim3(scatter_blocks),
                dim3(kReduceThreads),
                0,
                scatter_stream,
                static_cast<const T *>(gemm_output),
                static_cast<const T *>(expert_weight),
                static_cast<const float *>(topk_stage),
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                ntokens,
                n_dim,
                rank,
                group,
                chunk_size_n,
                this_chunk_size_n,
                signal_stride,
                epoch);
        }
        else
        {
            hipLaunchKernelGGL(
                (a2a_signal_scatter_one_group_kernel<T,
                                                     ApplyScale,
                                                     UseTopkStage>),
                dim3(scatter_blocks),
                dim3(kReduceThreads),
                0,
                scatter_stream,
                static_cast<const T *>(gemm_output),
                static_cast<const T *>(expert_weight),
                static_cast<const float *>(topk_stage),
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                group,
                chunk_size_n,
                signal_stride,
                epoch);
        }

        // The sum is also submitted without a resident polling CTA.  The
        // eight source-rank waits are ordered before this group's ordinary
        // sum launch and can therefore overlap the next AG/scatter group.
        for (int source_rank = 0; source_rank < world_size; ++source_rank)
        {
            void *signal = const_cast<void *>(
                signal_buffer_ptrs_host[source_rank]);
            CUDA_CHECK(hipStreamWaitValue32(
                sum_stream,
                static_cast<char *>(signal) +
                    static_cast<size_t>(signal_stride + group) * sizeof(int),
                static_cast<uint32_t>(epoch),
                hipStreamWaitValueGte,
                0xffffffffu));
        }
        if (use_wave64_sum && chunk_size_n == 512 &&
            this_chunk_size_n == 512)
        {
            hipLaunchKernelGGL((a2a_signal_sum_rank_wave64_512_kernel<T>),
                               dim3(sum_blocks),
                               dim3(kReduceThreads),
                               0,
                               sum_stream,
                               reduce_buffer_ptrs,
                               static_cast<T *>(output),
                               ntokens / world_size,
                               n_dim,
                               rank,
                               group);
        }
        else if (use_wave64_sum && chunk_size_n == 256 &&
                 this_chunk_size_n == 256)
        {
            hipLaunchKernelGGL((a2a_signal_sum_rank_wave64_256_kernel<T>),
                               dim3(sum_blocks),
                               dim3(kReduceThreads),
                               0,
                               sum_stream,
                               reduce_buffer_ptrs,
                               static_cast<T *>(output),
                               ntokens / world_size,
                               n_dim,
                               rank,
                               group);
        }
        else if (use_wave64_512x_sum)
        {
            hipLaunchKernelGGL((a2a_signal_sum_rank_wave64_512x_kernel<T>),
                               dim3(sum_blocks),
                               dim3(kReduceThreads),
                               0,
                               sum_stream,
                               reduce_buffer_ptrs,
                               static_cast<T *>(output),
                               ntokens / world_size,
                               n_dim,
                               rank,
                               group,
                               chunk_size_n,
                               this_chunk_size_n);
        }
        else
        {
            hipLaunchKernelGGL((a2a_signal_sum_rank_kernel<T, false>),
                               dim3(sum_blocks),
                               dim3(kReduceThreads),
                               0,
                               sum_stream,
                               reduce_buffer_ptrs,
                               signal_buffer_ptrs,
                               static_cast<T *>(output),
                               ntokens / world_size,
                               n_dim,
                               rank,
                               world_size,
                               chunk_size_n,
                               group,
                               group + 1,
                               signal_stride,
                               epoch);
        }
    }
}

template <typename T, bool ApplyScale>
void launch_a2a_signal_pipeline_hardware_wait_dtype(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    int early_scatter_block_budget,
    int early_sum_block_budget,
    int full_budget_start_group,
    bool wave64_512_scatter,
    bool wave64_512_sum,
    bool use_topk_stage,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    if (use_topk_stage)
    {
        launch_a2a_signal_pipeline_hardware_wait_impl<T, ApplyScale, true>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            scatter_block_budget,
            sum_block_budget,
            early_scatter_block_budget,
            early_sum_block_budget,
            full_budget_start_group,
            wave64_512_scatter,
            wave64_512_sum,
            scatter_stream,
            sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_hardware_wait_impl<T, ApplyScale, false>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            scatter_block_budget,
            sum_block_budget,
            early_scatter_block_budget,
            early_sum_block_budget,
            full_budget_start_group,
            wave64_512_scatter,
            wave64_512_sum,
            scatter_stream,
            sum_stream);
    }
}

template <typename T, bool ApplyScale>
void launch_a2a_signal_pipeline_stream_wait_dtype(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int block_budget,
    bool use_topk_stage,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    if (use_topk_stage)
    {
        launch_a2a_signal_pipeline_stream_wait_impl<T, ApplyScale, true>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            block_budget,
            scatter_stream,
            sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_stream_wait_impl<T, ApplyScale, false>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            output,
            ntokens,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            block_budget,
            scatter_stream,
            sum_stream);
    }
}

void moe_reduce_rs_a2a_signal_pipeline_stream_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    FTRAIN_HYGON_CHECK(dtype == c10::ScalarType::BFloat16 ||
               dtype == c10::ScalarType::Half)
        << "A2A stream-wait pipeline supports BF16/FP16 only";
    const bool apply_scale = expert_weight != nullptr;
    const bool use_topk_stage = topk_stage != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_stream_wait_dtype<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                signal_buffer_ptrs_host,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_stream_wait_dtype<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                signal_buffer_ptrs_host,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
    }
    else
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_stream_wait_dtype<__half, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                signal_buffer_ptrs_host,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_stream_wait_dtype<__half, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                signal_buffer_ptrs_host,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
    }
}

void moe_reduce_rs_a2a_signal_pipeline_hardware_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    int early_scatter_block_budget,
    int early_sum_block_budget,
    int full_budget_start_group,
    bool wave64_512_scatter,
    bool wave64_512_sum,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    FTRAIN_HYGON_CHECK(dtype == c10::ScalarType::BFloat16 ||
               dtype == c10::ScalarType::Half)
        << "A2A hardware-wait pipeline supports BF16/FP16 only";
    const bool apply_scale = expert_weight != nullptr;
    const bool use_topk_stage = topk_stage != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_hardware_wait_dtype<
                __hip_bfloat16,
                true>(gemm_output,
                      expert_weight,
                      topk_stage,
                      reduce_buffer_ptrs,
                      signal_buffer_ptrs,
                      signal_buffer_ptrs_host,
                      gemm_done_flags,
                      output,
                      ntokens,
                      n_dim,
                      rank,
                      world_size,
                      topk,
                      chunk_size_n,
                      num_groups,
                      signal_stride,
                      epoch,
                      scatter_block_budget,
                      sum_block_budget,
                      early_scatter_block_budget,
                      early_sum_block_budget,
                      full_budget_start_group,
                      wave64_512_scatter,
                      wave64_512_sum,
                      use_topk_stage,
                      scatter_stream,
                      sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_hardware_wait_dtype<
                __hip_bfloat16,
                false>(gemm_output,
                       expert_weight,
                       topk_stage,
                       reduce_buffer_ptrs,
                       signal_buffer_ptrs,
                       signal_buffer_ptrs_host,
                       gemm_done_flags,
                       output,
                       ntokens,
                       n_dim,
                       rank,
                       world_size,
                       topk,
                       chunk_size_n,
                       num_groups,
                       signal_stride,
                       epoch,
                       scatter_block_budget,
                       sum_block_budget,
                       early_scatter_block_budget,
                       early_sum_block_budget,
                       full_budget_start_group,
                       wave64_512_scatter,
                       wave64_512_sum,
                       use_topk_stage,
                       scatter_stream,
                       sum_stream);
        }
    }
    else
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_hardware_wait_dtype<__half, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                signal_buffer_ptrs_host,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                scatter_block_budget,
                sum_block_budget,
                early_scatter_block_budget,
                early_sum_block_budget,
                full_budget_start_group,
                wave64_512_scatter,
                wave64_512_sum,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_hardware_wait_dtype<__half, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                signal_buffer_ptrs_host,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                scatter_block_budget,
                sum_block_budget,
                early_scatter_block_budget,
                early_sum_block_budget,
                full_budget_start_group,
                wave64_512_scatter,
                wave64_512_sum,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
    }
}

void moe_reduce_rs_a2a_signal_pipeline_split_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int producer_block_budget,
    int consumer_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    FTRAIN_HYGON_CHECK(dtype == c10::ScalarType::BFloat16 ||
               dtype == c10::ScalarType::Half)
        << "A2A split pipeline supports BF16/FP16 only";
    const bool apply_scale = expert_weight != nullptr;
    const bool use_topk_stage = topk_stage != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_split_dtype<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                producer_block_budget,
                consumer_block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_split_dtype<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                producer_block_budget,
                consumer_block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
    }
    else
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_split_dtype<__half, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                producer_block_budget,
                consumer_block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_split_dtype<__half, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                producer_block_budget,
                consumer_block_budget,
                use_topk_stage,
                scatter_stream,
                sum_stream);
        }
    }
}

void moe_reduce_rs_a2a_signal_pipeline_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    FTRAIN_HYGON_CHECK(dtype == c10::ScalarType::BFloat16 ||
               dtype == c10::ScalarType::Half)
        << "A2A signal pipeline supports BF16/FP16 only";
    const bool apply_scale = expert_weight != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_dtype<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                topk_stage != nullptr,
                scatter_stream,
                sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_dtype<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                topk_stage != nullptr,
                scatter_stream,
                sum_stream);
        }
    }
    else
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_dtype<__half, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                topk_stage != nullptr,
                scatter_stream,
                sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_dtype<__half, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                gemm_done_flags,
                output,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                signal_stride,
                epoch,
                block_budget,
                topk_stage != nullptr,
                scatter_stream,
                sum_stream);
        }
    }
}


template <typename T, bool ApplyScale, bool UseTopkStage, int WorldSize>
void launch_a2a_signal_scatter_ragged_wave64_512x(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *rank_m_offsets,
    int ntokens,
    int n_dim,
    int rank,
    int topk,
    int group,
    int chunk_size_n,
    int this_chunk_size_n,
    int signal_stride,
    int epoch,
    int scatter_blocks,
    hipStream_t stream)
{
    if (topk == 6)
    {
        hipLaunchKernelGGL(
            (a2a_signal_scatter_one_group_ragged_wave64_512x_kernel<
                T, ApplyScale, UseTopkStage, 6, WorldSize>),
            dim3(scatter_blocks),
            dim3(kReduceThreads),
            0,
            stream,
            static_cast<const T *>(gemm_output),
            static_cast<const T *>(expert_weight),
            static_cast<const float *>(topk_stage),
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            rank_m_offsets,
            ntokens,
            n_dim,
            rank,
            topk,
            group,
            chunk_size_n,
            this_chunk_size_n,
            signal_stride,
            epoch);
    }
    else
    {
        hipLaunchKernelGGL(
            (a2a_signal_scatter_one_group_ragged_wave64_512x_kernel<
                T, ApplyScale, UseTopkStage, 0, WorldSize>),
            dim3(scatter_blocks),
            dim3(kReduceThreads),
            0,
            stream,
            static_cast<const T *>(gemm_output),
            static_cast<const T *>(expert_weight),
            static_cast<const float *>(topk_stage),
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            rank_m_offsets,
            ntokens,
            n_dim,
            rank,
            topk,
            group,
            chunk_size_n,
            this_chunk_size_n,
            signal_stride,
            epoch);
    }
}

// Persistent sibling of the helper above: one launch covers every N group and
// the kernel waits on the AG done flags itself.
template <typename T, bool ApplyScale, bool UseTopkStage, int WorldSize>
void launch_a2a_signal_scatter_ragged_wave64_512x_persistent(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *rank_m_offsets,
    const int *gemm_done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_blocks,
    hipStream_t stream)
{
    if (topk == 6)
    {
        hipLaunchKernelGGL(
            (a2a_signal_scatter_ragged_wave64_512x_persistent_kernel<
                T, ApplyScale, UseTopkStage, 6, WorldSize>),
            dim3(scatter_blocks),
            dim3(kReduceThreads),
            0,
            stream,
            static_cast<const T *>(gemm_output),
            static_cast<const T *>(expert_weight),
            static_cast<const float *>(topk_stage),
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            rank_m_offsets,
            gemm_done_flags,
            ntokens,
            n_dim,
            rank,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch);
    }
    else
    {
        hipLaunchKernelGGL(
            (a2a_signal_scatter_ragged_wave64_512x_persistent_kernel<
                T, ApplyScale, UseTopkStage, 0, WorldSize>),
            dim3(scatter_blocks),
            dim3(kReduceThreads),
            0,
            stream,
            static_cast<const T *>(gemm_output),
            static_cast<const T *>(expert_weight),
            static_cast<const float *>(topk_stage),
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            rank_m_offsets,
            gemm_done_flags,
            ntokens,
            n_dim,
            rank,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch);
    }
}


template <typename T, int WorldSize>
void launch_a2a_signal_sum_ragged_wave64_512x(
    void **reduce_buffer_ptrs,
    void *output,
    int local_rows,
    int n_dim,
    int rank,
    int group,
    int chunk_size_n,
    int this_chunk_size_n,
    int sum_blocks,
    hipStream_t stream)
{
    hipLaunchKernelGGL(
        (a2a_signal_sum_rank_ragged_wave64_512x_kernel<T, WorldSize>),
        dim3(sum_blocks),
        dim3(kReduceThreads),
        0,
        stream,
        reduce_buffer_ptrs,
        static_cast<T *>(output),
        local_rows,
        n_dim,
        rank,
        group,
        chunk_size_n,
        this_chunk_size_n);
}

template <typename T, bool ApplyScale, bool UseTopkStage>
void launch_a2a_signal_pipeline_ragged_hardware_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    const int scatter_budget =
        max(1, min(kReduceBlocks, scatter_block_budget));
    const int sum_budget = max(1, min(kReduceBlocks, sum_block_budget));
    const char *ragged_wave64_value =
        detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_RAGGED_WAVE64_512X");
    const bool ragged_wave64_enabled =
        ragged_wave64_value == nullptr ||
        (ragged_wave64_value[0] != '\0' && ragged_wave64_value[0] != '0' &&
         ragged_wave64_value[0] != 'f' && ragged_wave64_value[0] != 'F' &&
         ragged_wave64_value[0] != 'n' && ragged_wave64_value[0] != 'N');
    const bool ragged_wave64_contract =
        ragged_wave64_enabled && world_size >= 2 && world_size <= 8;

    auto launch_ragged_scatter = [&](int world,
                                      int group,
                                      int this_chunk_size_n,
                                      int scatter_blocks,
                                      int sum_blocks) {
        switch (world)
        {
        case 2:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 2>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 3:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 3>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 4:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 4>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 5:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 5>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 6:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 6>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 7:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 7>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 8:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 8>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        default:
            break;
        }
    };

    auto launch_ragged_sum = [&](int world,
                                  int group,
                                  int this_chunk_size_n,
                                  int scatter_blocks,
                                  int sum_blocks) {
        switch (world)
        {
        case 2:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 2>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 3:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 3>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 4:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 4>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 5:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 5>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 6:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 6>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 7:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 7>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 8:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 8>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        default:
            break;
        }
    };

    for (int group = 0; group < num_groups; ++group)
    {
        const int chunk_offset_n = group * chunk_size_n;
        const int this_chunk_size_n =
            min(chunk_size_n, n_dim - chunk_offset_n);
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const int64_t scatter_total =
            static_cast<int64_t>(ntokens) * this_chunk_size_n;
        const int64_t sum_total =
            static_cast<int64_t>(local_rows) * this_chunk_size_n;
        const int scatter_blocks =
            max(1, min(scatter_budget, capped_blocks(scatter_total)));
        const int sum_blocks =
            max(1, min(sum_budget, capped_blocks(sum_total)));
        const bool use_ragged_wave64_512x =
            ragged_wave64_contract && this_chunk_size_n >= 512 &&
            this_chunk_size_n % 512 == 0;

        CUDA_CHECK(hipStreamWaitValue32(
            scatter_stream,
            const_cast<int *>(gemm_done_flags + group),
            1u,
            hipStreamWaitValueGte,
            0xffffffffu));
        if (use_ragged_wave64_512x)
        {
            launch_ragged_scatter(world_size,
                                  group,
                                  this_chunk_size_n,
                                  scatter_blocks,
                                  sum_blocks);
        }
        else
        {
            hipLaunchKernelGGL(
                (a2a_signal_scatter_one_group_ragged_kernel<
                    T,
                    ApplyScale,
                    UseTopkStage>),
                dim3(scatter_blocks),
                dim3(kReduceThreads),
                0,
                scatter_stream,
                static_cast<const T *>(gemm_output),
                static_cast<const T *>(expert_weight),
                static_cast<const float *>(topk_stage),
                reduce_buffer_ptrs,
                signal_buffer_ptrs,
                rank_m_offsets,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                group,
                chunk_size_n,
                signal_stride,
                epoch);
        }
        CUDA_CHECK(hipGetLastError());

        for (int source_rank = 0; source_rank < world_size; ++source_rank)
        {
            void *signal = const_cast<void *>(
                signal_buffer_ptrs_host[source_rank]);
            CUDA_CHECK(hipStreamWaitValue32(
                sum_stream,
                static_cast<char *>(signal) +
                    static_cast<size_t>(signal_stride + group) * sizeof(int),
                static_cast<uint32_t>(epoch),
                hipStreamWaitValueGte,
                0xffffffffu));
        }
        if (use_ragged_wave64_512x)
        {
            launch_ragged_sum(world_size,
                              group,
                              this_chunk_size_n,
                              scatter_blocks,
                              sum_blocks);
        }
        else
        {
            hipLaunchKernelGGL(
                (a2a_signal_sum_rank_ragged_one_group_kernel<T>),
                dim3(sum_blocks),
                dim3(kReduceThreads),
                0,
                sum_stream,
                reduce_buffer_ptrs,
                static_cast<T *>(output),
                local_rows,
                n_dim,
                rank,
                world_size,
                group,
                chunk_size_n);
        }
        CUDA_CHECK(hipGetLastError());
    }
}

template <typename T, bool ApplyScale>
void launch_a2a_signal_pipeline_ragged_hardware_wait_dtype(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    bool use_topk_stage,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    if (use_topk_stage)
    {
        launch_a2a_signal_pipeline_ragged_hardware_wait_impl<
            T,
            ApplyScale,
            true>(gemm_output,
                  expert_weight,
                  topk_stage,
                  reduce_buffer_ptrs,
                  signal_buffer_ptrs,
                  signal_buffer_ptrs_host,
                  gemm_done_flags,
                  rank_m_offsets,
                  output,
                  ntokens,
                  local_rows,
                  n_dim,
                  rank,
                  world_size,
                  topk,
                  chunk_size_n,
                  num_groups,
                  signal_stride,
                  epoch,
                  scatter_block_budget,
                  sum_block_budget,
                  scatter_stream,
                  sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_ragged_hardware_wait_impl<
            T,
            ApplyScale,
            false>(gemm_output,
                   expert_weight,
                   topk_stage,
                   reduce_buffer_ptrs,
                   signal_buffer_ptrs,
                   signal_buffer_ptrs_host,
                   gemm_done_flags,
                   rank_m_offsets,
                   output,
                   ntokens,
                   local_rows,
                   n_dim,
                   rank,
                   world_size,
                   topk,
                   chunk_size_n,
                   num_groups,
                   signal_stride,
                   epoch,
                   scatter_block_budget,
                   sum_block_budget,
                   scatter_stream,
                   sum_stream);
    }
}

void moe_reduce_rs_a2a_signal_pipeline_ragged_hardware_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    FTRAIN_HYGON_CHECK(dtype == c10::ScalarType::BFloat16 ||
               dtype == c10::ScalarType::Half)
        << "Ragged A2A hardware-wait pipeline supports BF16/FP16 only";
    FTRAIN_HYGON_CHECK(local_rows > 0)
        << "Ragged A2A requires a positive local destination M";
    const bool apply_scale = expert_weight != nullptr;
    const bool use_topk_stage = topk_stage != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_ragged_hardware_wait_dtype<
                __hip_bfloat16,
                true>(gemm_output,
                      expert_weight,
                      topk_stage,
                      reduce_buffer_ptrs,
                      signal_buffer_ptrs,
                      signal_buffer_ptrs_host,
                      gemm_done_flags,
                      rank_m_offsets,
                      output,
                      ntokens,
                      local_rows,
                      n_dim,
                      rank,
                      world_size,
                      topk,
                      chunk_size_n,
                      num_groups,
                      signal_stride,
                      epoch,
                      scatter_block_budget,
                      sum_block_budget,
                      use_topk_stage,
                      scatter_stream,
                      sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_ragged_hardware_wait_dtype<
                __hip_bfloat16,
                false>(gemm_output,
                       expert_weight,
                       topk_stage,
                       reduce_buffer_ptrs,
                       signal_buffer_ptrs,
                       signal_buffer_ptrs_host,
                       gemm_done_flags,
                       rank_m_offsets,
                       output,
                       ntokens,
                       local_rows,
                       n_dim,
                       rank,
                       world_size,
                       topk,
                       chunk_size_n,
                       num_groups,
                       signal_stride,
                       epoch,
                       scatter_block_budget,
                       sum_block_budget,
                       use_topk_stage,
                       scatter_stream,
                       sum_stream);
        }
    }
    else if (apply_scale)
    {
        launch_a2a_signal_pipeline_ragged_hardware_wait_dtype<__half, true>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            rank_m_offsets,
            output,
            ntokens,
            local_rows,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            scatter_block_budget,
            sum_block_budget,
            use_topk_stage,
            scatter_stream,
            sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_ragged_hardware_wait_dtype<__half, false>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            rank_m_offsets,
            output,
            ntokens,
            local_rows,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            scatter_block_budget,
            sum_block_budget,
            use_topk_stage,
            scatter_stream,
            sum_stream);
    }
}

template <typename T>
void launch_moe_reduce_rs_compute(const void *input,
                                  const void *weight,
                                  const void *chosen_experts,
                                  const void *expert_weight,
                                  void *partial,
                                  int ntokens,
                                  int n_dim,
                                  int k_per_rank,
                                  int num_experts,
                                  int topk,
                                  c10::ScalarType index_dtype,
                                  hipStream_t stream)
{
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(ntokens) * n_dim;
    const dim3 grid((total + threads - 1) / threads);
    const dim3 block(threads);
    if (index_dtype == c10::ScalarType::Long)
    {
        hipLaunchKernelGGL((moe_reduce_rs_compute_kernel<T, int64_t>),
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const T *>(input),
                           static_cast<const T *>(weight),
                           chosen_experts,
                           static_cast<const T *>(expert_weight),
                           static_cast<T *>(partial),
                           ntokens,
                           n_dim,
                           k_per_rank,
                           num_experts,
                           topk);
    }
    else
    {
        hipLaunchKernelGGL((moe_reduce_rs_compute_kernel<T, int32_t>),
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const T *>(input),
                           static_cast<const T *>(weight),
                           chosen_experts,
                           static_cast<const T *>(expert_weight),
                           static_cast<T *>(partial),
                           ntokens,
                           n_dim,
                           k_per_rank,
                           num_experts,
                           topk);
    }
}

template <typename T>
void launch_moe_reduce_rs_sum(void **partial_ptrs,
                              void *output,
                              int world_size,
                              int rank,
                              int ntokens,
                              int n_dim,
                              hipStream_t stream)
{
    const int threads = 256;
    const int tokens_per_rank = ntokens / world_size;
    const int64_t total = static_cast<int64_t>(tokens_per_rank) * n_dim;
    hipLaunchKernelGGL(moe_reduce_rs_sum_kernel<T>,
                       dim3((total + threads - 1) / threads),
                       dim3(threads),
                       0,
                       stream,
                       partial_ptrs,
                       static_cast<T *>(output),
                       world_size,
                       rank,
                       ntokens,
                       n_dim);
}

} // namespace

void moe_reduce_rs_a2a_signal_pipeline(const void *gemm_output,
                                       const void *expert_weight,
                                       const void *topk_stage,
                                       void **reduce_buffer_ptrs,
                                       void **signal_buffer_ptrs,
                                       const int *gemm_done_flags,
                                       void *output,
                                       int ntokens,
                                       int n_dim,
                                       int rank,
                                       int world_size,
                                       int topk,
                                       int chunk_size_n,
                                       int num_groups,
                                       int signal_stride,
                                       int epoch,
                                       int block_budget,
                                       c10::ScalarType dtype,
                                       hipStream_t scatter_stream,
                                       hipStream_t sum_stream)
{
    moe_reduce_rs_a2a_signal_pipeline_impl(gemm_output,
                                           expert_weight,
                                           topk_stage,
                                           reduce_buffer_ptrs,
                                           signal_buffer_ptrs,
                                           gemm_done_flags,
                                           output,
                                           ntokens,
                                           n_dim,
                                           rank,
                                           world_size,
                                           topk,
                                           chunk_size_n,
                                           num_groups,
                                           signal_stride,
                                           epoch,
                                           block_budget,
                                           dtype,
                                           scatter_stream,
                                           sum_stream);
}

void moe_reduce_rs_a2a_signal_pipeline_split(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int producer_block_budget,
    int consumer_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    moe_reduce_rs_a2a_signal_pipeline_split_impl(
        gemm_output,
        expert_weight,
        topk_stage,
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        gemm_done_flags,
        output,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch,
        producer_block_budget,
        consumer_block_budget,
        dtype,
        scatter_stream,
        sum_stream);
}

void moe_reduce_rs_a2a_signal_pipeline_stream_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    moe_reduce_rs_a2a_signal_pipeline_stream_wait_impl(
        gemm_output,
        expert_weight,
        topk_stage,
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        signal_buffer_ptrs_host,
        gemm_done_flags,
        output,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch,
        block_budget,
        dtype,
        scatter_stream,
        sum_stream);
}

void moe_reduce_rs_a2a_signal_pipeline_hardware_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    void *output,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    int early_scatter_block_budget,
    int early_sum_block_budget,
    int full_budget_start_group,
    bool wave64_512_scatter,
    bool wave64_512_sum,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    moe_reduce_rs_a2a_signal_pipeline_hardware_wait_impl(
        gemm_output,
        expert_weight,
        topk_stage,
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        signal_buffer_ptrs_host,
        gemm_done_flags,
        output,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch,
        scatter_block_budget,
        sum_block_budget,
        early_scatter_block_budget,
        early_sum_block_budget,
        full_budget_start_group,
        wave64_512_scatter,
        wave64_512_sum,
        dtype,
        scatter_stream,
        sum_stream);
}

void moe_reduce_rs_a2a_signal_pipeline_ragged_hardware_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    moe_reduce_rs_a2a_signal_pipeline_ragged_hardware_wait_impl(
        gemm_output,
        expert_weight,
        topk_stage,
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        signal_buffer_ptrs_host,
        gemm_done_flags,
        rank_m_offsets,
        output,
        ntokens,
        local_rows,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch,
        scatter_block_budget,
        sum_block_budget,
        dtype,
        scatter_stream,
        sum_stream);
}
template <typename T, bool ApplyScale, bool UseTopkStage>
void launch_a2a_signal_pipeline_ragged_stream_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    const int scatter_budget =
        max(1, min(kReduceBlocks, scatter_block_budget));
    const int sum_budget = max(1, min(kReduceBlocks, sum_block_budget));
    const char *ragged_wave64_value =
        detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_RAGGED_WAVE64_512X");
    const bool ragged_wave64_enabled =
        ragged_wave64_value == nullptr ||
        (ragged_wave64_value[0] != '\0' && ragged_wave64_value[0] != '0' &&
         ragged_wave64_value[0] != 'f' && ragged_wave64_value[0] != 'F' &&
         ragged_wave64_value[0] != 'n' && ragged_wave64_value[0] != 'N');
    const bool ragged_wave64_contract =
        ragged_wave64_enabled && world_size >= 2 && world_size <= 8;

    auto launch_ragged_scatter = [&](int world,
                                      int group,
                                      int this_chunk_size_n,
                                      int scatter_blocks,
                                      int sum_blocks) {
        switch (world)
        {
        case 2:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 2>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 3:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 3>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 4:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 4>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 5:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 5>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 6:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 6>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 7:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 7>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        case 8:
            launch_a2a_signal_scatter_ragged_wave64_512x<
                T, ApplyScale, UseTopkStage, 8>(
                gemm_output, expert_weight, topk_stage,
                reduce_buffer_ptrs, signal_buffer_ptrs, rank_m_offsets,
                ntokens, n_dim, rank, topk, group, chunk_size_n,
                this_chunk_size_n, signal_stride, epoch, scatter_blocks,
                scatter_stream);
            break;
        default:
            break;
        }
    };

    auto launch_ragged_sum = [&](int world,
                                  int group,
                                  int this_chunk_size_n,
                                  int scatter_blocks,
                                  int sum_blocks) {
        switch (world)
        {
        case 2:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 2>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 3:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 3>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 4:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 4>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 5:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 5>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 6:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 6>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 7:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 7>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        case 8:
            launch_a2a_signal_sum_ragged_wave64_512x<T, 8>(
                reduce_buffer_ptrs, output, local_rows, n_dim, rank,
                group, chunk_size_n, this_chunk_size_n, sum_blocks,
                sum_stream);
            break;
        default:
            break;
        }
    };

    // Stream-wait variant: the scatter side is one persistent producer that
    // loops over every N group and spins on the AG done flags in-kernel; only
    // the sum side is split into one ordinary launch per group (below).
    const int wave64_tail =
        (n_dim % chunk_size_n == 0) ? 0 : (n_dim % chunk_size_n);
    const char *persistent_wave64_value =
        detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_RAGGED_WAVE64_PERSISTENT");
    const bool persistent_wave64_enabled =
        persistent_wave64_value != nullptr &&
        persistent_wave64_value[0] != '\0' &&
        persistent_wave64_value[0] != '0' &&
        persistent_wave64_value[0] != 'f' && persistent_wave64_value[0] != 'F' &&
        persistent_wave64_value[0] != 'n' && persistent_wave64_value[0] != 'N';
    const bool persistent_wave64_ok =
        persistent_wave64_enabled && ragged_wave64_contract &&
        chunk_size_n >= 512 &&
        chunk_size_n % 512 == 0 &&
        (wave64_tail == 0 || (wave64_tail >= 512 && wave64_tail % 512 == 0));
    const int64_t persistent_scatter_total =
        static_cast<int64_t>(ntokens) * n_dim;
    const int persistent_scatter_blocks =
        max(1, min(scatter_budget, capped_blocks(persistent_scatter_total)));
    if (persistent_wave64_ok)
    {
        switch (world_size)
        {
        case 2:
            launch_a2a_signal_scatter_ragged_wave64_512x_persistent<
                T, ApplyScale, UseTopkStage, 2>(
                gemm_output, expert_weight, topk_stage, reduce_buffer_ptrs,
                signal_buffer_ptrs, rank_m_offsets, gemm_done_flags, ntokens,
                n_dim, rank, topk, chunk_size_n, num_groups, signal_stride,
                epoch, persistent_scatter_blocks, scatter_stream);
            break;
        case 3:
            launch_a2a_signal_scatter_ragged_wave64_512x_persistent<
                T, ApplyScale, UseTopkStage, 3>(
                gemm_output, expert_weight, topk_stage, reduce_buffer_ptrs,
                signal_buffer_ptrs, rank_m_offsets, gemm_done_flags, ntokens,
                n_dim, rank, topk, chunk_size_n, num_groups, signal_stride,
                epoch, persistent_scatter_blocks, scatter_stream);
            break;
        case 4:
            launch_a2a_signal_scatter_ragged_wave64_512x_persistent<
                T, ApplyScale, UseTopkStage, 4>(
                gemm_output, expert_weight, topk_stage, reduce_buffer_ptrs,
                signal_buffer_ptrs, rank_m_offsets, gemm_done_flags, ntokens,
                n_dim, rank, topk, chunk_size_n, num_groups, signal_stride,
                epoch, persistent_scatter_blocks, scatter_stream);
            break;
        case 5:
            launch_a2a_signal_scatter_ragged_wave64_512x_persistent<
                T, ApplyScale, UseTopkStage, 5>(
                gemm_output, expert_weight, topk_stage, reduce_buffer_ptrs,
                signal_buffer_ptrs, rank_m_offsets, gemm_done_flags, ntokens,
                n_dim, rank, topk, chunk_size_n, num_groups, signal_stride,
                epoch, persistent_scatter_blocks, scatter_stream);
            break;
        case 6:
            launch_a2a_signal_scatter_ragged_wave64_512x_persistent<
                T, ApplyScale, UseTopkStage, 6>(
                gemm_output, expert_weight, topk_stage, reduce_buffer_ptrs,
                signal_buffer_ptrs, rank_m_offsets, gemm_done_flags, ntokens,
                n_dim, rank, topk, chunk_size_n, num_groups, signal_stride,
                epoch, persistent_scatter_blocks, scatter_stream);
            break;
        case 7:
            launch_a2a_signal_scatter_ragged_wave64_512x_persistent<
                T, ApplyScale, UseTopkStage, 7>(
                gemm_output, expert_weight, topk_stage, reduce_buffer_ptrs,
                signal_buffer_ptrs, rank_m_offsets, gemm_done_flags, ntokens,
                n_dim, rank, topk, chunk_size_n, num_groups, signal_stride,
                epoch, persistent_scatter_blocks, scatter_stream);
            break;
        case 8:
            launch_a2a_signal_scatter_ragged_wave64_512x_persistent<
                T, ApplyScale, UseTopkStage, 8>(
                gemm_output, expert_weight, topk_stage, reduce_buffer_ptrs,
                signal_buffer_ptrs, rank_m_offsets, gemm_done_flags, ntokens,
                n_dim, rank, topk, chunk_size_n, num_groups, signal_stride,
                epoch, persistent_scatter_blocks, scatter_stream);
            break;
        default:
            break;
        }
    }
    else
    {
    hipLaunchKernelGGL(
        (a2a_signal_scatter_ragged_kernel<T, ApplyScale, UseTopkStage>),
        dim3(persistent_scatter_blocks),
        dim3(kReduceThreads),
        0,
        scatter_stream,
        static_cast<const T *>(gemm_output),
        static_cast<const T *>(expert_weight),
        static_cast<const float *>(topk_stage),
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        rank_m_offsets,
        gemm_done_flags,
        ntokens,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch);
    CUDA_CHECK(hipGetLastError());
    }

    for (int group = 0; group < num_groups; ++group)
    {
        const int chunk_offset_n = group * chunk_size_n;
        const int this_chunk_size_n =
            min(chunk_size_n, n_dim - chunk_offset_n);
        if (this_chunk_size_n <= 0)
        {
            continue;
        }

        const int64_t scatter_total =
            static_cast<int64_t>(ntokens) * this_chunk_size_n;
        const int64_t sum_total =
            static_cast<int64_t>(local_rows) * this_chunk_size_n;
        const int scatter_blocks =
            max(1, min(scatter_budget, capped_blocks(scatter_total)));
        const int sum_blocks =
            max(1, min(sum_budget, capped_blocks(sum_total)));
        const bool use_ragged_wave64_512x =
            ragged_wave64_contract && this_chunk_size_n >= 512 &&
            this_chunk_size_n % 512 == 0;

        for (int source_rank = 0; source_rank < world_size; ++source_rank)
        {
            void *signal = const_cast<void *>(
                signal_buffer_ptrs_host[source_rank]);
            CUDA_CHECK(hipStreamWaitValue32(
                sum_stream,
                static_cast<char *>(signal) +
                    static_cast<size_t>(signal_stride + group) * sizeof(int),
                static_cast<uint32_t>(epoch),
                hipStreamWaitValueGte,
                0xffffffffu));
        }
        if (use_ragged_wave64_512x)
        {
            launch_ragged_sum(world_size,
                              group,
                              this_chunk_size_n,
                              scatter_blocks,
                              sum_blocks);
        }
        else
        {
            hipLaunchKernelGGL(
                (a2a_signal_sum_rank_ragged_one_group_kernel<T>),
                dim3(sum_blocks),
                dim3(kReduceThreads),
                0,
                sum_stream,
                reduce_buffer_ptrs,
                static_cast<T *>(output),
                local_rows,
                n_dim,
                rank,
                world_size,
                group,
                chunk_size_n);
        }
        CUDA_CHECK(hipGetLastError());
    }
}
template <typename T, bool ApplyScale>
void launch_a2a_signal_pipeline_ragged_stream_wait_dtype(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    bool use_topk_stage,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    if (use_topk_stage)
    {
        launch_a2a_signal_pipeline_ragged_stream_wait_impl<
            T,
            ApplyScale,
            true>(gemm_output,
                  expert_weight,
                  topk_stage,
                  reduce_buffer_ptrs,
                  signal_buffer_ptrs,
                  signal_buffer_ptrs_host,
                  gemm_done_flags,
                  rank_m_offsets,
                  output,
                  ntokens,
                  local_rows,
                  n_dim,
                  rank,
                  world_size,
                  topk,
                  chunk_size_n,
                  num_groups,
                  signal_stride,
                  epoch,
                  scatter_block_budget,
                  sum_block_budget,
                  scatter_stream,
                  sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_ragged_stream_wait_impl<
            T,
            ApplyScale,
            false>(gemm_output,
                   expert_weight,
                   topk_stage,
                   reduce_buffer_ptrs,
                   signal_buffer_ptrs,
                   signal_buffer_ptrs_host,
                   gemm_done_flags,
                   rank_m_offsets,
                   output,
                   ntokens,
                   local_rows,
                   n_dim,
                   rank,
                   world_size,
                   topk,
                   chunk_size_n,
                   num_groups,
                   signal_stride,
                   epoch,
                   scatter_block_budget,
                   sum_block_budget,
                   scatter_stream,
                   sum_stream);
    }
}
void moe_reduce_rs_a2a_signal_pipeline_ragged_stream_wait_impl(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    FTRAIN_HYGON_CHECK(dtype == c10::ScalarType::BFloat16 ||
               dtype == c10::ScalarType::Half)
        << "Ragged A2A hardware-wait pipeline supports BF16/FP16 only";
    FTRAIN_HYGON_CHECK(local_rows > 0)
        << "Ragged A2A requires a positive local destination M";
    const bool apply_scale = expert_weight != nullptr;
    const bool use_topk_stage = topk_stage != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_a2a_signal_pipeline_ragged_stream_wait_dtype<
                __hip_bfloat16,
                true>(gemm_output,
                      expert_weight,
                      topk_stage,
                      reduce_buffer_ptrs,
                      signal_buffer_ptrs,
                      signal_buffer_ptrs_host,
                      gemm_done_flags,
                      rank_m_offsets,
                      output,
                      ntokens,
                      local_rows,
                      n_dim,
                      rank,
                      world_size,
                      topk,
                      chunk_size_n,
                      num_groups,
                      signal_stride,
                      epoch,
                      scatter_block_budget,
                      sum_block_budget,
                      use_topk_stage,
                      scatter_stream,
                      sum_stream);
        }
        else
        {
            launch_a2a_signal_pipeline_ragged_stream_wait_dtype<
                __hip_bfloat16,
                false>(gemm_output,
                       expert_weight,
                       topk_stage,
                       reduce_buffer_ptrs,
                       signal_buffer_ptrs,
                       signal_buffer_ptrs_host,
                       gemm_done_flags,
                       rank_m_offsets,
                       output,
                       ntokens,
                       local_rows,
                       n_dim,
                       rank,
                       world_size,
                       topk,
                       chunk_size_n,
                       num_groups,
                       signal_stride,
                       epoch,
                       scatter_block_budget,
                       sum_block_budget,
                       use_topk_stage,
                       scatter_stream,
                       sum_stream);
        }
    }
    else if (apply_scale)
    {
        launch_a2a_signal_pipeline_ragged_stream_wait_dtype<__half, true>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            rank_m_offsets,
            output,
            ntokens,
            local_rows,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            scatter_block_budget,
            sum_block_budget,
            use_topk_stage,
            scatter_stream,
            sum_stream);
    }
    else
    {
        launch_a2a_signal_pipeline_ragged_stream_wait_dtype<__half, false>(
            gemm_output,
            expert_weight,
            topk_stage,
            reduce_buffer_ptrs,
            signal_buffer_ptrs,
            signal_buffer_ptrs_host,
            gemm_done_flags,
            rank_m_offsets,
            output,
            ntokens,
            local_rows,
            n_dim,
            rank,
            world_size,
            topk,
            chunk_size_n,
            num_groups,
            signal_stride,
            epoch,
            scatter_block_budget,
            sum_block_budget,
            use_topk_stage,
            scatter_stream,
            sum_stream);
    }
}

void moe_reduce_rs_a2a_signal_pipeline_ragged_stream_wait(
    const void *gemm_output,
    const void *expert_weight,
    const void *topk_stage,
    void **reduce_buffer_ptrs,
    void **signal_buffer_ptrs,
    const void *const *signal_buffer_ptrs_host,
    const int *gemm_done_flags,
    const int *rank_m_offsets,
    void *output,
    int ntokens,
    int local_rows,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_stride,
    int epoch,
    int scatter_block_budget,
    int sum_block_budget,
    c10::ScalarType dtype,
    hipStream_t scatter_stream,
    hipStream_t sum_stream)
{
    moe_reduce_rs_a2a_signal_pipeline_ragged_stream_wait_impl(
        gemm_output,
        expert_weight,
        topk_stage,
        reduce_buffer_ptrs,
        signal_buffer_ptrs,
        signal_buffer_ptrs_host,
        gemm_done_flags,
        rank_m_offsets,
        output,
        ntokens,
        local_rows,
        n_dim,
        rank,
        world_size,
        topk,
        chunk_size_n,
        num_groups,
        signal_stride,
        epoch,
        scatter_block_budget,
        sum_block_budget,
        dtype,
        scatter_stream,
        sum_stream);
}


void moe_reduce_rs_prepare_counts(const void *chosen_experts,
                                  int *expert_counts,
                                  int ntokens,
                                  int num_experts,
                                  int topk,
                                  bool index_is_i64,
                                  hipStream_t stream)
{
    const int total_routes = ntokens * topk;
    const dim3 grid(capped_blocks(total_routes));
    const dim3 block(kReduceThreads);
    if (index_is_i64)
    {
        hipLaunchKernelGGL(prepare_counts_kernel<int64_t>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const int64_t *>(chosen_experts),
                           expert_counts,
                           total_routes,
                           num_experts);
    }
    else
    {
        hipLaunchKernelGGL(prepare_counts_kernel<int32_t>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const int32_t *>(chosen_experts),
                           expert_counts,
                           total_routes,
                           num_experts);
    }
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_set_done_flags(int *done_flags,
                                  int group,
                                  hipStream_t stream)
{
    hipLaunchKernelGGL(set_done_flag_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       done_flags,
                       group);
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_scale_output(void *output,
                                const void *scale,
                                int rows,
                                int n_dim,
                                c10::ScalarType dtype,
                                hipStream_t stream)
{
    const int64_t total = static_cast<int64_t>(rows) * n_dim;
    const dim3 grid(capped_work_blocks(total));
    const dim3 block(kReduceThreads);
    if (dtype == c10::ScalarType::BFloat16)
    {
        hipLaunchKernelGGL(scale_output_kernel<__hip_bfloat16>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<__hip_bfloat16 *>(output),
                           static_cast<const __hip_bfloat16 *>(scale),
                           rows,
                           n_dim);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        hipLaunchKernelGGL(scale_output_kernel<__half>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<__half *>(output),
                           static_cast<const __half *>(scale),
                           rows,
                           n_dim);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS scale only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

template <typename T>
void launch_te_moe_unpermute_topk_reduce(
    const void *packed_output,
    const int32_t *inverse_sorted_rows,
    const void *expert_weight,
    void *partial,
    int ntokens,
    int n_dim,
    int topk,
    hipStream_t stream)
{
    const bool use_pack8 = (n_dim & 7) == 0 &&
                           (reinterpret_cast<uintptr_t>(packed_output) & 15) ==
                               0 &&
                           (reinterpret_cast<uintptr_t>(partial) & 15) == 0;
    const int64_t total = static_cast<int64_t>(ntokens) *
                          (use_pack8 ? n_dim / 8 : n_dim);
    const dim3 grid(capped_work_blocks(total));
    const dim3 block(kReduceThreads);

#define FTRAIN_HYGON_LAUNCH_TE_UNPERMUTE(TOPK)                                      \
    if (use_pack8)                                                          \
    {                                                                       \
        hipLaunchKernelGGL(                                                 \
            (te_moe_unpermute_topk_reduce_pack8_kernel<T, TOPK>),           \
            grid,                                                           \
            block,                                                          \
            0,                                                              \
            stream,                                                         \
            static_cast<const T *>(packed_output),                          \
            inverse_sorted_rows,                                            \
            static_cast<const T *>(expert_weight),                          \
            static_cast<T *>(partial),                                      \
            ntokens,                                                        \
            n_dim,                                                          \
            topk);                                                          \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        hipLaunchKernelGGL(                                                 \
            (te_moe_unpermute_topk_reduce_scalar_kernel<T, TOPK>),          \
            grid,                                                           \
            block,                                                          \
            0,                                                              \
            stream,                                                         \
            static_cast<const T *>(packed_output),                          \
            inverse_sorted_rows,                                            \
            static_cast<const T *>(expert_weight),                          \
            static_cast<T *>(partial),                                      \
            ntokens,                                                        \
            n_dim,                                                          \
            topk);                                                          \
    }

    switch (topk)
    {
    case 2:
        FTRAIN_HYGON_LAUNCH_TE_UNPERMUTE(2);
        break;
    case 4:
        FTRAIN_HYGON_LAUNCH_TE_UNPERMUTE(4);
        break;
    case 6:
        FTRAIN_HYGON_LAUNCH_TE_UNPERMUTE(6);
        break;
    case 8:
        FTRAIN_HYGON_LAUNCH_TE_UNPERMUTE(8);
        break;
    default:
        FTRAIN_HYGON_LAUNCH_TE_UNPERMUTE(0);
        break;
    }
#undef FTRAIN_HYGON_LAUNCH_TE_UNPERMUTE
}

void te_moe_unpermute_topk_reduce(const void *packed_output,
                                  const int32_t *inverse_sorted_rows,
                                  const void *expert_weight,
                                  void *partial,
                                  int ntokens,
                                  int n_dim,
                                  int topk,
                                  c10::ScalarType dtype,
                                  hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(packed_output != nullptr && inverse_sorted_rows != nullptr &&
               expert_weight != nullptr && partial != nullptr);
    FTRAIN_HYGON_CHECK(ntokens > 0 && n_dim > 0 && topk > 0);
    if (dtype == c10::ScalarType::BFloat16)
    {
        launch_te_moe_unpermute_topk_reduce<__hip_bfloat16>(packed_output,
                                                             inverse_sorted_rows,
                                                             expert_weight,
                                                             partial,
                                                             ntokens,
                                                             n_dim,
                                                             topk,
                                                             stream);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        launch_te_moe_unpermute_topk_reduce<__half>(packed_output,
                                                    inverse_sorted_rows,
                                                    expert_weight,
                                                    partial,
                                                    ntokens,
                                                    n_dim,
                                                    topk,
                                                    stream);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false)
            << "TE MoE fused postprocess only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_build_topk_stage(
    const void *gemm_output,
    const void *expert_weight,
    void *topk_stage,
    const int *gemm_done_flags,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    int ntokens,
    int n_dim,
    int topk,
    int chunk_size_n,
    int num_groups,
    int signal_value,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(gemm_output != nullptr && expert_weight != nullptr &&
               topk_stage != nullptr);
    FTRAIN_HYGON_CHECK(ntokens > 0 && n_dim > 0 && topk > 0 && chunk_size_n > 0);
    FTRAIN_HYGON_CHECK(num_groups == (n_dim + chunk_size_n - 1) / chunk_size_n);
    FTRAIN_HYGON_CHECK((n_chunk_done_counts == nullptr) ==
               (n_chunk_done_flags == nullptr));
    FTRAIN_HYGON_CHECK(gemm_done_flags == nullptr ||
               (n_chunk_done_counts != nullptr && n_chunk_done_flags != nullptr));

    const int64_t max_group_total =
        static_cast<int64_t>(ntokens) * std::min(chunk_size_n, n_dim);
    const dim3 grid(capped_work_blocks(max_group_total), num_groups);
    const dim3 block(kReduceThreads);
    if (dtype == c10::ScalarType::BFloat16)
    {
        hipLaunchKernelGGL(build_topk_stage_chunked_kernel<__hip_bfloat16>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const __hip_bfloat16 *>(gemm_output),
                           static_cast<const __hip_bfloat16 *>(expert_weight),
                           static_cast<float *>(topk_stage),
                           gemm_done_flags,
                           n_chunk_done_counts,
                           n_chunk_done_flags,
                           ntokens,
                           n_dim,
                           topk,
                           chunk_size_n,
                           signal_value);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        hipLaunchKernelGGL(build_topk_stage_chunked_kernel<__half>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const __half *>(gemm_output),
                           static_cast<const __half *>(expert_weight),
                           static_cast<float *>(topk_stage),
                           gemm_done_flags,
                           n_chunk_done_counts,
                           n_chunk_done_flags,
                           ntokens,
                           n_dim,
                           topk,
                           chunk_size_n,
                           signal_value);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false)
            << "MoEReduceRS topk-stage build only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}



void moe_reduce_rs_topk_reduce_a2a(const void *gemm_output,
                                   const void *expert_weight,
                                   void **reduce_buffer_ptrs,
                                   int *gemm_done_flags,
                                   int ntokens,
                                   int n_dim,
                                   int rank,
                                   int world_size,
                                   int topk,
                                   int chunk_size_n,
                                   int num_groups,
                                   c10::ScalarType dtype,
                                   hipStream_t stream)
{
    const bool apply_scale = expert_weight != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_groups<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                0,
                num_groups,
                0,
                stream);
        }
        else
        {
            launch_topk_scatter_a2a_groups<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                0,
                num_groups,
                0,
                stream);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_groups<__half, true>(gemm_output,
                                                         expert_weight,
                                                         reduce_buffer_ptrs,
                                                         gemm_done_flags,
                                                         ntokens,
                                                         n_dim,
                                                         rank,
                                                         world_size,
                                                         topk,
                                                         chunk_size_n,
                                                         0,
                                                         num_groups,
                                                         0,
                                                         stream);
        }
        else
        {
            launch_topk_scatter_a2a_groups<__half, false>(gemm_output,
                                                          expert_weight,
                                                          reduce_buffer_ptrs,
                                                          gemm_done_flags,
                                                          ntokens,
                                                          n_dim,
                                                          rank,
                                                          world_size,
                                                          topk,
                                                          chunk_size_n,
                                                          0,
                                                          num_groups,
                                                          0,
                                                          stream);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS A2A only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_topk_reduce_a2a_range(const void *gemm_output,
                                         const void *expert_weight,
                                         void **reduce_buffer_ptrs,
                                         int *gemm_done_flags,
                                         int ntokens,
                                         int n_dim,
                                         int rank,
                                         int world_size,
                                         int topk,
                                         int chunk_size_n,
                                         int first_group,
                                         int group_count,
                                         int block_budget,
                                         c10::ScalarType dtype,
                                         hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(first_group >= 0 && group_count > 0 && block_budget > 0);
    const bool apply_scale = expert_weight != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_groups<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                first_group,
                group_count,
                block_budget,
                stream);
        }
        else
        {
            launch_topk_scatter_a2a_groups<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                first_group,
                group_count,
                block_budget,
                stream);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_groups<__half, true>(gemm_output,
                                                         expert_weight,
                                                         reduce_buffer_ptrs,
                                                         gemm_done_flags,
                                                         ntokens,
                                                         n_dim,
                                                         rank,
                                                         world_size,
                                                         topk,
                                                         chunk_size_n,
                                                         first_group,
                                                         group_count,
                                                         block_budget,
                                                         stream);
        }
        else
        {
            launch_topk_scatter_a2a_groups<__half, false>(gemm_output,
                                                          expert_weight,
                                                          reduce_buffer_ptrs,
                                                          gemm_done_flags,
                                                          ntokens,
                                                          n_dim,
                                                          rank,
                                                          world_size,
                                                          topk,
                                                          chunk_size_n,
                                                          first_group,
                                                          group_count,
                                                          block_budget,
                                                          stream);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS A2A only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_topk_reduce_a2a_paced_range(
    const void *gemm_output,
    const void *expert_weight,
    void **reduce_buffer_ptrs,
    int *gemm_done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int first_group,
    int group_count,
    int block_budget,
    int check_iterations,
    int sleep_steps,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(first_group >= 0 && group_count > 0 && block_budget > 0);
    FTRAIN_HYGON_CHECK(first_group + group_count <= num_groups);
    FTRAIN_HYGON_CHECK(check_iterations > 0 && sleep_steps >= 0);
    const bool apply_scale = expert_weight != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_paced_groups<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                first_group,
                group_count,
                block_budget,
                check_iterations,
                sleep_steps,
                stream);
        }
        else
        {
            launch_topk_scatter_a2a_paced_groups<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                first_group,
                group_count,
                block_budget,
                check_iterations,
                sleep_steps,
                stream);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_paced_groups<__half, true>(gemm_output,
                                                               expert_weight,
                                                               reduce_buffer_ptrs,
                                                               gemm_done_flags,
                                                               ntokens,
                                                               n_dim,
                                                               rank,
                                                               world_size,
                                                               topk,
                                                               chunk_size_n,
                                                               num_groups,
                                                               first_group,
                                                               group_count,
                                                               block_budget,
                                                               check_iterations,
                                                               sleep_steps,
                                                               stream);
        }
        else
        {
            launch_topk_scatter_a2a_paced_groups<__half, false>(gemm_output,
                                                                expert_weight,
                                                                reduce_buffer_ptrs,
                                                                gemm_done_flags,
                                                                ntokens,
                                                                n_dim,
                                                                rank,
                                                                world_size,
                                                                topk,
                                                                chunk_size_n,
                                                                num_groups,
                                                                first_group,
                                                                group_count,
                                                                block_budget,
                                                                check_iterations,
                                                                sleep_steps,
                                                                stream);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS paced A2A only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

template <typename T, bool ApplyScale>
void launch_topk_scatter_a2a_admitted_impl(const void *gemm_output,
                                           const void *expert_weight,
                                           void **reduce_buffer_ptrs,
                                           int *gemm_done_flags,
                                           int ntokens,
                                           int n_dim,
                                           int rank,
                                           int world_size,
                                           int topk,
                                           int chunk_size_n,
                                           int num_groups,
                                           int batch_groups,
                                           int early_block_budget,
                                           int final_block_budget,
                                           hipStream_t stream)
{
    const int early_group_count = max(0, num_groups - 1);
    for (int first_group = 0; first_group < early_group_count;
         first_group += batch_groups)
    {
        const int group_count =
            min(batch_groups, early_group_count - first_group);
        CUDA_CHECK(hipStreamWaitValue32(stream,
                                        gemm_done_flags + first_group,
                                        1u,
                                        hipStreamWaitValueGte,
                                        0xffffffffu));
        launch_topk_scatter_a2a_groups<T, ApplyScale>(gemm_output,
                                                      expert_weight,
                                                      reduce_buffer_ptrs,
                                                      gemm_done_flags,
                                                      ntokens,
                                                      n_dim,
                                                      rank,
                                                      world_size,
                                                      topk,
                                                      chunk_size_n,
                                                      first_group,
                                                      group_count,
                                                      early_block_budget,
                                                      stream);
    }
    const int final_group = num_groups - 1;
    CUDA_CHECK(hipStreamWaitValue32(stream,
                                    gemm_done_flags + final_group,
                                    1u,
                                    hipStreamWaitValueGte,
                                    0xffffffffu));
    launch_topk_scatter_a2a_groups<T, ApplyScale>(gemm_output,
                                                  expert_weight,
                                                  reduce_buffer_ptrs,
                                                  gemm_done_flags,
                                                  ntokens,
                                                  n_dim,
                                                  rank,
                                                  world_size,
                                                  topk,
                                                  chunk_size_n,
                                                  final_group,
                                                  1,
                                                  final_block_budget,
                                                  stream);
}

void moe_reduce_rs_topk_reduce_a2a_admitted(
    const void *gemm_output,
    const void *expert_weight,
    void **reduce_buffer_ptrs,
    int *gemm_done_flags,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int batch_groups,
    int early_block_budget,
    int final_block_budget,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(num_groups > 0 && batch_groups > 0 && early_block_budget > 0 &&
               final_block_budget > 0);
    const bool apply_scale = expert_weight != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_admitted_impl<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                batch_groups,
                early_block_budget,
                final_block_budget,
                stream);
        }
        else
        {
            launch_topk_scatter_a2a_admitted_impl<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                batch_groups,
                early_block_budget,
                final_block_budget,
                stream);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_admitted_impl<__half, true>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                batch_groups,
                early_block_budget,
                final_block_budget,
                stream);
        }
        else
        {
            launch_topk_scatter_a2a_admitted_impl<__half, false>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                batch_groups,
                early_block_budget,
                final_block_budget,
                stream);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS admitted A2A only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

template <typename T, bool ApplyScale>
void launch_topk_scatter_a2a_adaptive_admitted_impl(
    const void *gemm_output,
    const void *expert_weight,
    void **reduce_buffer_ptrs,
    int *gemm_done_flags,
    int *completion_counts,
    int problem_count,
    int physical_n_tiles,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int lead_groups,
    int lead_percent,
    int early_block_budget,
    int final_block_budget,
    hipStream_t stream)
{
    const int physical_tiles_per_group = chunk_size_n / 256;
    const int final_group = num_groups - 1;
    for (int group = 0; group < final_group; ++group)
    {
        const int admission_group = min(final_group, group + lead_groups);
        const int first_physical_tile =
            admission_group * physical_tiles_per_group;
        const int admission_tiles = min(
            physical_tiles_per_group,
            max(0, physical_n_tiles - first_physical_tile));
        const int expected = problem_count * admission_tiles;
        const int threshold = max(
            1,
            static_cast<int>((static_cast<int64_t>(expected) * lead_percent +
                              99) /
                             100));
        CUDA_CHECK(hipStreamWaitValue32(stream,
                                        completion_counts + admission_group,
                                        static_cast<uint32_t>(threshold),
                                        hipStreamWaitValueGte,
                                        0xffffffffu));
        launch_topk_scatter_a2a_groups<T, ApplyScale>(gemm_output,
                                                      expert_weight,
                                                      reduce_buffer_ptrs,
                                                      gemm_done_flags,
                                                      ntokens,
                                                      n_dim,
                                                      rank,
                                                      world_size,
                                                      topk,
                                                      chunk_size_n,
                                                      group,
                                                      1,
                                                      early_block_budget,
                                                      stream);
    }

    CUDA_CHECK(hipStreamWaitValue32(stream,
                                    gemm_done_flags + final_group,
                                    1u,
                                    hipStreamWaitValueGte,
                                    0xffffffffu));
    launch_topk_scatter_a2a_groups<T, ApplyScale>(gemm_output,
                                                  expert_weight,
                                                  reduce_buffer_ptrs,
                                                  gemm_done_flags,
                                                  ntokens,
                                                  n_dim,
                                                  rank,
                                                  world_size,
                                                  topk,
                                                  chunk_size_n,
                                                  final_group,
                                                  1,
                                                  final_block_budget,
                                                  stream);
}

void moe_reduce_rs_topk_reduce_a2a_adaptive_admitted(
    const void *gemm_output,
    const void *expert_weight,
    void **reduce_buffer_ptrs,
    int *gemm_done_flags,
    int *completion_counts,
    int problem_count,
    int physical_n_tiles,
    int ntokens,
    int n_dim,
    int rank,
    int world_size,
    int topk,
    int chunk_size_n,
    int num_groups,
    int lead_groups,
    int lead_percent,
    int early_block_budget,
    int final_block_budget,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(gemm_done_flags != nullptr && completion_counts != nullptr);
    FTRAIN_HYGON_CHECK(problem_count > 0 && physical_n_tiles > 0 && num_groups >= 2);
    FTRAIN_HYGON_CHECK(chunk_size_n > 0 && chunk_size_n % 256 == 0);
    FTRAIN_HYGON_CHECK(lead_groups > 0 && lead_percent > 0 && lead_percent <= 100);
    FTRAIN_HYGON_CHECK(early_block_budget > 0 && final_block_budget > 0);

    const bool apply_scale = expert_weight != nullptr;
#define FTRAIN_HYGON_LAUNCH_ADAPTIVE_ADMITTED(T, APPLY_SCALE)                         \
    launch_topk_scatter_a2a_adaptive_admitted_impl<T, APPLY_SCALE>(          \
        gemm_output,                                                         \
        expert_weight,                                                       \
        reduce_buffer_ptrs,                                                  \
        gemm_done_flags,                                                     \
        completion_counts,                                                   \
        problem_count,                                                       \
        physical_n_tiles,                                                    \
        ntokens,                                                             \
        n_dim,                                                               \
        rank,                                                                \
        world_size,                                                          \
        topk,                                                                \
        chunk_size_n,                                                        \
        num_groups,                                                          \
        lead_groups,                                                         \
        lead_percent,                                                        \
        early_block_budget,                                                  \
        final_block_budget,                                                  \
        stream)

    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            FTRAIN_HYGON_LAUNCH_ADAPTIVE_ADMITTED(__hip_bfloat16, true);
        }
        else
        {
            FTRAIN_HYGON_LAUNCH_ADAPTIVE_ADMITTED(__hip_bfloat16, false);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (apply_scale)
        {
            FTRAIN_HYGON_LAUNCH_ADAPTIVE_ADMITTED(__half, true);
        }
        else
        {
            FTRAIN_HYGON_LAUNCH_ADAPTIVE_ADMITTED(__half, false);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false)
            << "MoEReduceRS adaptive-admitted A2A only supports FP16/BF16";
    }
#undef FTRAIN_HYGON_LAUNCH_ADAPTIVE_ADMITTED
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_topk_reduce_a2a_chunk(const void *gemm_output,
                                         const void *expert_weight,
                                         void **reduce_buffer_ptrs,
                                         int *gemm_done_flags,
                                         int ntokens,
                                         int n_dim,
                                         int rank,
                                         int world_size,
                                         int topk,
                                         int group,
                                         int chunk_offset_n,
                                         int chunk_size_n,
                                         c10::ScalarType dtype,
                                         hipStream_t stream)
{
    const bool apply_scale = expert_weight != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_chunk<__hip_bfloat16, true>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                group,
                chunk_offset_n,
                chunk_size_n,
                stream);
        }
        else
        {
            launch_topk_scatter_a2a_chunk<__hip_bfloat16, false>(
                gemm_output,
                expert_weight,
                reduce_buffer_ptrs,
                gemm_done_flags,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                group,
                chunk_offset_n,
                chunk_size_n,
                stream);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (apply_scale)
        {
            launch_topk_scatter_a2a_chunk<__half, true>(gemm_output,
                                                        expert_weight,
                                                        reduce_buffer_ptrs,
                                                        gemm_done_flags,
                                                        ntokens,
                                                        n_dim,
                                                        rank,
                                                        world_size,
                                                        topk,
                                                        group,
                                                        chunk_offset_n,
                                                        chunk_size_n,
                                                        stream);
        }
        else
        {
            launch_topk_scatter_a2a_chunk<__half, false>(gemm_output,
                                                         expert_weight,
                                                         reduce_buffer_ptrs,
                                                         gemm_done_flags,
                                                         ntokens,
                                                         n_dim,
                                                         rank,
                                                         world_size,
                                                         topk,
                                                         group,
                                                         chunk_offset_n,
                                                         chunk_size_n,
                                                         stream);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS chunk A2A only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_topk_stage_scatter_a2a(const void *topk_stage,
                                          void **reduce_buffer_ptrs,
                                          int *gemm_done_flags,
                                          int ntokens,
                                          int n_dim,
                                          int rank,
                                          int world_size,
                                          int chunk_size_n,
                                          int num_groups,
                                          c10::ScalarType dtype,
                                          hipStream_t stream)
{
    const int64_t largest_group =
        static_cast<int64_t>(ntokens) * chunk_size_n;
    const dim3 grid(capped_blocks(largest_group));
    const dim3 block(kReduceThreads);
    if (dtype == c10::ScalarType::BFloat16)
    {
        hipLaunchKernelGGL(topk_stage_scatter_a2a_kernel<__hip_bfloat16>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const float *>(topk_stage),
                           reduce_buffer_ptrs,
                           gemm_done_flags,
                           ntokens,
                           n_dim,
                           rank,
                           world_size,
                           chunk_size_n,
                           num_groups);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        hipLaunchKernelGGL(topk_stage_scatter_a2a_kernel<__half>,
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const float *>(topk_stage),
                           reduce_buffer_ptrs,
                           gemm_done_flags,
                           ntokens,
                           n_dim,
                           rank,
                           world_size,
                           chunk_size_n,
                           num_groups);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS topk-stage only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_a2a_cooperative_fused(const void *gemm_output,
                                         const void *expert_weight,
                                         const void *topk_stage,
                                         void **reduce_buffer_ptrs,
                                         int **sync_buffer_ptrs,
                                         int *grid_barrier,
                                         int *gemm_done_flags,
                                         void *output,
                                         const int *rank_m_offsets,
                                         int local_rows,
                                         int ntokens,
                                         int n_dim,
                                         int rank,
                                         int world_size,
                                         int topk,
                                         int chunk_size_n,
                                         int num_groups,
                                         int epoch,
                                         c10::ScalarType dtype,
                                         bool hip_cooperative_launch,
                                         bool group_pipeline,
                                         hipStream_t stream)
{
    FTRAIN_HYGON_CHECK((gemm_output != nullptr) != (topk_stage != nullptr))
        << "fused A2A requires exactly one of gemm_output/topk_stage";
    FTRAIN_HYGON_CHECK(reduce_buffer_ptrs != nullptr &&
               sync_buffer_ptrs != nullptr && grid_barrier != nullptr &&
               gemm_done_flags != nullptr && output != nullptr);
    FTRAIN_HYGON_CHECK(world_size >= 2 && world_size <= kReduceThreads);
    FTRAIN_HYGON_CHECK(ntokens > 0 && n_dim > 0 && topk > 0 &&
               chunk_size_n > 0 && num_groups > 0);
    const bool ragged = rank_m_offsets != nullptr;
    FTRAIN_HYGON_CHECK(ragged || ntokens % world_size == 0)
        << "fixed cooperative A2A requires equal rank M";
    FTRAIN_HYGON_CHECK(!ragged || local_rows > 0)
        << "ragged cooperative A2A requires positive local M";
    FTRAIN_HYGON_CHECK(!ragged || !group_pipeline)
        << "ragged cooperative A2A does not support group_pipeline";
    const bool use_topk_stage = topk_stage != nullptr;
    const bool apply_scale = expert_weight != nullptr;
    if (dtype == c10::ScalarType::BFloat16)
    {
        if (use_topk_stage)
        {
            launch_a2a_cooperative_fused_dispatch<__hip_bfloat16, false, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                sync_buffer_ptrs,
                grid_barrier,
                gemm_done_flags,
                output,
                rank_m_offsets,
                local_rows,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                epoch,
                group_pipeline,
                hip_cooperative_launch,
                stream);
        }
        else if (apply_scale)
        {
            launch_a2a_cooperative_fused_dispatch<__hip_bfloat16, true, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                sync_buffer_ptrs,
                grid_barrier,
                gemm_done_flags,
                output,
                rank_m_offsets,
                local_rows,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                epoch,
                group_pipeline,
                hip_cooperative_launch,
                stream);
        }
        else
        {
            launch_a2a_cooperative_fused_dispatch<__hip_bfloat16, false, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                sync_buffer_ptrs,
                grid_barrier,
                gemm_done_flags,
                output,
                rank_m_offsets,
                local_rows,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                epoch,
                group_pipeline,
                hip_cooperative_launch,
                stream);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (use_topk_stage)
        {
            launch_a2a_cooperative_fused_dispatch<__half, false, true>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                sync_buffer_ptrs,
                grid_barrier,
                gemm_done_flags,
                output,
                rank_m_offsets,
                local_rows,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                epoch,
                group_pipeline,
                hip_cooperative_launch,
                stream);
        }
        else if (apply_scale)
        {
            launch_a2a_cooperative_fused_dispatch<__half, true, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                sync_buffer_ptrs,
                grid_barrier,
                gemm_done_flags,
                output,
                rank_m_offsets,
                local_rows,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                epoch,
                group_pipeline,
                hip_cooperative_launch,
                stream);
        }
        else
        {
            launch_a2a_cooperative_fused_dispatch<__half, false, false>(
                gemm_output,
                expert_weight,
                topk_stage,
                reduce_buffer_ptrs,
                sync_buffer_ptrs,
                grid_barrier,
                gemm_done_flags,
                output,
                rank_m_offsets,
                local_rows,
                ntokens,
                n_dim,
                rank,
                world_size,
                topk,
                chunk_size_n,
                num_groups,
                epoch,
                group_pipeline,
                hip_cooperative_launch,
                stream);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "fused A2A only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

void moe_reduce_rs_sum_a2a_chunk(void **reduce_buffer_ptrs,
                                 void *output,
                                 int ntokens_per_rank,
                                 int n_dim,
                                 int rank,
                                 int world_size,
                                 int chunk_offset_n,
                                 int chunk_size_n,
                                 c10::ScalarType dtype,
                                 hipStream_t stream)
{
    const int64_t total =
        static_cast<int64_t>(ntokens_per_rank) * chunk_size_n;
    const dim3 grid(capped_blocks(total));
    const dim3 block(kReduceThreads);
    if (dtype == c10::ScalarType::BFloat16)
    {
        hipLaunchKernelGGL(sum_a2a_source_kernel<__hip_bfloat16>,
                           grid,
                           block,
                           0,
                           stream,
                           reduce_buffer_ptrs,
                           static_cast<__hip_bfloat16 *>(output),
                           ntokens_per_rank,
                           n_dim,
                           rank,
                           world_size,
                           chunk_offset_n,
                           chunk_size_n);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        hipLaunchKernelGGL(sum_a2a_source_kernel<__half>,
                           grid,
                           block,
                           0,
                           stream,
                           reduce_buffer_ptrs,
                           static_cast<__half *>(output),
                           ntokens_per_rank,
                           n_dim,
                           rank,
                           world_size,
                           chunk_offset_n,
                           chunk_size_n);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "MoEReduceRS sum A2A only supports FP16/BF16";
    }
    CUDA_CHECK(hipGetLastError());
}

template <typename T, int WorldSize>
void launch_sum_a2a_source_wave64_512x(void **reduce_buffer_ptrs,
                                       void *output,
                                       int ntokens_per_rank,
                                       int n_dim,
                                       int rank,
                                       hipStream_t stream)
{
    const int64_t total =
        static_cast<int64_t>(ntokens_per_rank) * n_dim;
    int sum_blocks = capped_blocks(total);
    const char *sum_blocks_value =
        detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_ORDINARY_SUM_BLOCKS");
    if (sum_blocks_value != nullptr && *sum_blocks_value != '\0')
    {
        const int requested = std::atoi(sum_blocks_value);
        if (requested > 0)
        {
            sum_blocks = std::min(capped_work_blocks(total), requested);
        }
    }
    hipLaunchKernelGGL((sum_a2a_source_wave64_512x_kernel<T, WorldSize>),
                       dim3(sum_blocks),
                       dim3(kReduceThreads),
                       0,
                       stream,
                       reduce_buffer_ptrs,
                       static_cast<T *>(output),
                       ntokens_per_rank,
                       n_dim,
                       rank);
}

void moe_reduce_rs_sum_a2a(void **reduce_buffer_ptrs,
                           void *output,
                           int ntokens_per_rank,
                           int n_dim,
                           int rank,
                           int world_size,
                           c10::ScalarType dtype,
                           hipStream_t stream)
{
    const char *wave64_value =
        detail::get_vendor_env("FTRAIN_HYGON_MOE_RS_A2A_ORDINARY_WAVE64_512X_SUM");
    const bool wave64_enabled =
        wave64_value == nullptr || std::atoi(wave64_value) != 0;
    const bool use_wave64_512x =
        wave64_enabled && world_size >= 1 && world_size <= 8 &&
        n_dim % 512 == 0;
    if (use_wave64_512x && dtype == c10::ScalarType::BFloat16)
    {
        switch (world_size)
        {
        case 2:
            launch_sum_a2a_source_wave64_512x<__hip_bfloat16, 2>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 3:
            launch_sum_a2a_source_wave64_512x<__hip_bfloat16, 3>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 4:
            launch_sum_a2a_source_wave64_512x<__hip_bfloat16, 4>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 5:
            launch_sum_a2a_source_wave64_512x<__hip_bfloat16, 5>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 6:
            launch_sum_a2a_source_wave64_512x<__hip_bfloat16, 6>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 7:
            launch_sum_a2a_source_wave64_512x<__hip_bfloat16, 7>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 8:
            launch_sum_a2a_source_wave64_512x<__hip_bfloat16, 8>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        default:
            break;
        }
    }
    else if (use_wave64_512x && dtype == c10::ScalarType::Half)
    {
        switch (world_size)
        {
        case 2:
            launch_sum_a2a_source_wave64_512x<__half, 2>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 3:
            launch_sum_a2a_source_wave64_512x<__half, 3>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 4:
            launch_sum_a2a_source_wave64_512x<__half, 4>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 5:
            launch_sum_a2a_source_wave64_512x<__half, 5>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 6:
            launch_sum_a2a_source_wave64_512x<__half, 6>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 7:
            launch_sum_a2a_source_wave64_512x<__half, 7>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        case 8:
            launch_sum_a2a_source_wave64_512x<__half, 8>(
                reduce_buffer_ptrs, output, ntokens_per_rank, n_dim, rank,
                stream);
            return;
        default:
            break;
        }
    }

    moe_reduce_rs_sum_a2a_chunk(reduce_buffer_ptrs,
                                output,
                                ntokens_per_rank,
                                n_dim,
                                rank,
                                world_size,
                                0,
                                n_dim,
                                dtype,
                                stream);
}







void moe_reduce_rs_recovered_compute_impl(const void *input,
                                          const void *weight,
                                          const void *chosen_experts,
                                          const void *expert_weight,
                                          void *partial,
                                          int ntokens,
                                          int n_dim,
                                          int k_per_rank,
                                          int num_experts,
                                          int topk,
                                          c10::ScalarType dtype,
                                          c10::ScalarType index_dtype,
                                          hipStream_t stream)
{
    if (dtype == c10::ScalarType::BFloat16)
    {
        launch_moe_reduce_rs_compute<__hip_bfloat16>(input,
                                                     weight,
                                                     chosen_experts,
                                                     expert_weight,
                                                     partial,
                                                     ntokens,
                                                     n_dim,
                                                     k_per_rank,
                                                     num_experts,
                                                     topk,
                                                     index_dtype,
                                                     stream);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        launch_moe_reduce_rs_compute<__half>(input,
                                             weight,
                                             chosen_experts,
                                             expert_weight,
                                             partial,
                                             ntokens,
                                             n_dim,
                                             k_per_rank,
                                             num_experts,
                                             topk,
                                             index_dtype,
                                             stream);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "Recovered MoEReduceRS only supports FP16/BF16";
    }
}

void moe_reduce_rs_recovered_sum_impl(void **partial_ptrs,
                                      void *output,
                                      int world_size,
                                      int rank,
                                      int ntokens,
                                      int n_dim,
                                      c10::ScalarType dtype,
                                      hipStream_t stream)
{
    if (dtype == c10::ScalarType::BFloat16)
    {
        launch_moe_reduce_rs_sum<__hip_bfloat16>(
            partial_ptrs, output, world_size, rank, ntokens, n_dim, stream);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        launch_moe_reduce_rs_sum<__half>(
            partial_ptrs, output, world_size, rank, ntokens, n_dim, stream);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "Recovered MoEReduceRS only supports FP16/BF16";
    }
}

} // namespace hygon_vendor
} // namespace bytedance
