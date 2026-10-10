#include "hygon_vendor.h"
#include "hip/hip_bfloat16.h"
#include "hip/hip_ext.h"
#include "hip/hip_fp16.h"
#include "hip/hip_runtime.h"
#include "hip_kernels/ag_group_gemm_kernels.h"
#include <c10/core/ScalarType.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <type_traits>
#include <stdint.h>

namespace bytedance
{
namespace hygon_vendor
{
namespace
{

constexpr int AG_RANK_TILE_INFO_STRIDE = 5;
constexpr int AG_RANK_TILE_EXPERT = 0;
constexpr int AG_RANK_TILE_SORTED_START = 1;
constexpr int AG_RANK_TILE_ROWS = 2;
constexpr int AG_RANK_TILE_RANK_START = 3;
constexpr int AG_RANK_TILE_RANK_END = 4;
constexpr int AG_RANK_TILE_PROBLEM_INFO_STRIDE = 8;
constexpr int AG_RANK_TILE_PROBLEM_CHUNK_START = 5;
constexpr int AG_RANK_TILE_PROBLEM_CHUNK_END = 6;
constexpr int AG_RANK_TILE_PROBLEM_RESERVED = 7;

struct AgRankMOffsets
{
    int32_t values[MAX_LOCAL_WORLD_SIZE + 1];
};

bool get_env_bool_local(const char *name, bool default_value)
{
    const char *value = detail::get_vendor_env(name);
    if (value == nullptr)
    {
        return default_value;
    }
    return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

int get_env_int_local(const char *name, int default_value)
{
    const char *value = detail::get_vendor_env(name);
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

const char *get_env_with_legacy(const char *name, const char *legacy_name)
{
    const char *value = detail::get_vendor_env(name);
    return value != nullptr ? value : detail::get_vendor_env(legacy_name);
}

bool get_env_bool_with_legacy(const char *name,
                              const char *legacy_name,
                              bool default_value)
{
    const char *value = get_env_with_legacy(name, legacy_name);
    if (value == nullptr)
    {
        return default_value;
    }
    return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
}

int get_env_int_with_legacy(const char *name,
                            const char *legacy_name,
                            int default_value)
{
    const char *value = get_env_with_legacy(name, legacy_name);
    if (value == nullptr || value[0] == '\0')
    {
        return default_value;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end == value ? default_value : static_cast<int>(parsed);
}

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

using ag_half4_t =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using ag_bf16x4_t =
    __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
using ag_bf16x8_t =
    __attribute__((__vector_size__(8 * sizeof(uint16_t)))) uint16_t;
using ag_float4_t =
    __attribute__((__vector_size__(4 * sizeof(float)))) float;
using ag_uint4_t =
    __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;

__device__ inline uint16_t ag_float_to_bf16_bits(float value)
{
    uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>(bits >> 16);
}

__device__ inline ag_float4_t ag_mmac_f16(ag_half4_t a,
                                          ag_half4_t b,
                                          ag_float4_t c)
{
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16f16(a, b, c);
#else
    return c;
#endif
}

__device__ inline void ag_ds_read_m32x16_b16(ag_uint4_t &dst,
                                             uint32_t byte_offset)
{
    asm volatile("ds_read_m32x16_b16 %0 %1\n\t"
                 : "=v"(dst)
                 : "v"(byte_offset)
                 : "memory");
}

__device__ inline void ag_wait_lgkmcnt0()
{
    asm volatile("s_waitcnt lgkmcnt(0)\n\t" ::: "memory");
}

__device__ inline ag_half4_t ag_half4_from_u32x2(const ag_uint4_t &src,
                                                 int pair)
{
    ag_half4_t out;
    const uint32_t lo = src[pair * 2 + 0];
    const uint32_t hi = src[pair * 2 + 1];
    reinterpret_cast<uint32_t *>(&out)[0] = lo;
    reinterpret_cast<uint32_t *>(&out)[1] = hi;
    return out;
}

__device__ inline ag_bf16x4_t ag_bf16x4_from_u32x2(const ag_uint4_t &src,
                                                   int pair)
{
    ag_bf16x4_t out;
    const uint32_t lo = src[pair * 2 + 0];
    const uint32_t hi = src[pair * 2 + 1];
    reinterpret_cast<uint32_t *>(&out)[0] = lo;
    reinterpret_cast<uint32_t *>(&out)[1] = hi;
    return out;
}

__device__ inline ag_float4_t ag_mmac_bf16(ag_bf16x4_t a,
                                           ag_bf16x4_t b,
                                           ag_float4_t c)
{
#if defined(__gfx936__) || defined(__gfx938__) || defined(__gfx928__)
    return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
#else
    return c;
#endif
}

__device__ inline int ag_dsread_m32x16_lane_offset_elems(int lane)
{
    return (lane >> 2) * 32 + (lane & 3) * 8;
}

__device__ inline int ag_dsread_m32x16_b16_fragment_slot(int row16, int k32)
{
    const int kg = k32 & 3;
    const int half = k32 >> 4;
    const int inner = (k32 >> 2) & 3;
    return kg * 128 + half * 16 + inner * 32 + row16;
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_col(int row16, int col32)
{
    constexpr int kVec = 8;
    constexpr int kPerPhase = 4;
    constexpr int kMaxPhase = 2;
    const int phase = (row16 / kPerPhase) % kMaxPhase;
    return ((col32 / kVec) ^ phase) * kVec + (col32 & (kVec - 1));
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_slot(int row16, int col32)
{
    return row16 * 32 +
           ag_dsread_m32x16_b16_swizzled_col(row16, col32);
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_physical_slot(int slot)
{
    const int row = slot >> 5;
    const int col = slot & 31;
    return row * 32 + ag_dsread_m32x16_b16_swizzled_col(row, col);
}

__device__ inline int ag_dsread_m32x16_b16_swizzled_lane_offset_elems(int lane)
{
    return ag_dsread_m32x16_b16_swizzled_physical_slot(
        ag_dsread_m32x16_lane_offset_elems(lane));
}

__device__ inline void ag_wait_signal_ge(const int *signals,
                                         int index,
                                         int signal_value)
{
    const volatile int *signal =
        reinterpret_cast<const volatile int *>(signals + index);
    while (*signal < signal_value)
    {
        __builtin_amdgcn_s_sleep(1);
    }
}

__device__ inline int ag_token_chunk_id(int token,
                                        int full_m,
                                        int world_size,
                                        int signal_chunk_rows,
                                        int chunks_per_rank)
{
    const int local_m = full_m / world_size;
    int owner_rank = token / local_m;
    owner_rank = owner_rank < 0 ? 0 : owner_rank;
    owner_rank = owner_rank >= world_size ? world_size - 1 : owner_rank;
    int local_row = token - owner_rank * local_m;
    local_row = local_row < 0 ? 0 : local_row;
    int chunk = local_row / signal_chunk_rows;
    chunk = chunk < 0 ? 0 : chunk;
    chunk = chunk >= chunks_per_rank ? chunks_per_rank - 1 : chunk;
    return owner_rank * chunks_per_rank + chunk;
}

template <typename T>
__device__ inline void zero_vec4(T &value)
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = 0;
    }
}

__device__ inline ag_half4_t load_f16x4_k_contiguous(const _Float16 *ptr,
                                                     int64_t base,
                                                     int k,
                                                     int k_dim)
{
    ag_half4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim) ? ptr[base + i] : static_cast<_Float16>(0);
    }
    return value;
}

__device__ inline ag_half4_t load_f16x4_weight_kn(const _Float16 *weight,
                                                  int expert,
                                                  int k,
                                                  int n,
                                                  int k_dim,
                                                  int n_dim)
{
    ag_half4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim)
                       ? weight[(static_cast<int64_t>(expert) * k_dim + k + i) *
                                    n_dim +
                                n]
                       : static_cast<_Float16>(0);
    }
    return value;
}

__device__ inline ag_half4_t load_f16x4_weight_nk(const _Float16 *weight,
                                                  int expert,
                                                  int k,
                                                  int n,
                                                  int k_dim,
                                                  int n_dim)
{
    ag_half4_t value;
    const int64_t base =
        (static_cast<int64_t>(expert) * n_dim + n) * k_dim + k;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim)
                       ? weight[base + i]
                       : static_cast<_Float16>(0);
    }
    return value;
}

__device__ inline ag_half4_t load_f16x4_weight_shared_bn(
    const _Float16 *sh_b,
    int k,
    int n,
    int bn)
{
    ag_half4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = sh_b[(k + i) * bn + n];
    }
    return value;
}

__device__ inline ag_bf16x4_t load_bf16x4_k_contiguous(const uint16_t *ptr,
                                                       int64_t base,
                                                       int k,
                                                       int k_dim)
{
    ag_bf16x4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim) ? ptr[base + i] : 0;
    }
    return value;
}

__device__ inline ag_bf16x4_t load_bf16x4_weight_kn(const uint16_t *weight,
                                                    int expert,
                                                    int k,
                                                    int n,
                                                    int k_dim,
                                                    int n_dim)
{
    ag_bf16x4_t value;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim)
                       ? weight[(static_cast<int64_t>(expert) * k_dim + k + i) *
                                    n_dim +
                                n]
                       : 0;
    }
    return value;
}

__device__ inline ag_bf16x4_t load_bf16x4_weight_nk(const uint16_t *weight,
                                                    int expert,
                                                    int k,
                                                    int n,
                                                    int k_dim,
                                                    int n_dim)
{
    ag_bf16x4_t value;
    const int64_t base =
        (static_cast<int64_t>(expert) * n_dim + n) * k_dim + k;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        value[i] = (k + i < k_dim && n < n_dim) ? weight[base + i] : 0;
    }
    return value;
}

template <typename IndexT>
__device__ inline int load_index(const void *ptr, int offset)
{
    return static_cast<int>(static_cast<const IndexT *>(ptr)[offset]);
}

__global__ void update_rank_m_offsets_kernel(int *device_offsets,
                                             AgRankMOffsets offsets,
                                             int count)
{
    const int index = threadIdx.x;
    if (index < count)
    {
        device_offsets[index] = offsets.values[index];
    }
}

template <typename T, typename IndexT>
__global__ void ag_group_gemm_reference_kernel(const T *full_input,
                                               const T *weight,
                                               const void *topk_ids,
                                               T *output,
                                               int full_m,
                                               int n_dim,
                                               int k_dim,
                                               int num_experts,
                                               int topk)
{
    const int64_t total = static_cast<int64_t>(full_m) * topk * n_dim;
    const int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total)
    {
        return;
    }

    const int n = linear % n_dim;
    const int flat_row = linear / n_dim;
    const int token = flat_row / topk;
    const int expert = load_index<IndexT>(topk_ids, flat_row);
    if (expert < 0 || expert >= num_experts)
    {
        output[linear] = store_from_float<T>(0.0f);
        return;
    }

    float acc = 0.0f;
    const T *x = full_input + static_cast<int64_t>(token) * k_dim;
    const T *w = weight + (static_cast<int64_t>(expert) * k_dim * n_dim) + n;
    for (int k = 0; k < k_dim; ++k)
    {
        acc += load_as_float<T>(x[k]) *
               load_as_float<T>(w[static_cast<int64_t>(k) * n_dim]);
    }
    output[linear] = store_from_float<T>(acc);
}

__device__ inline int ag_push_stage_for_owner(int owner_rank,
                                              int current_rank,
                                              int world_size,
                                              bool zigzag_order)
{
    const int forward =
        (owner_rank - current_rank + world_size) % world_size;
    if (!zigzag_order || forward == 0)
    {
        return forward;
    }
    const int backward = world_size - forward;
    return forward <= backward ? 2 * forward - 1 : 2 * backward;
}

__device__ inline int ag_owner_rank_from_offsets(int token,
                                                 const int *rank_m_offsets,
                                                 int world_size)
{
#pragma unroll
    for (int owner_rank = 0; owner_rank < MAX_LOCAL_WORLD_SIZE; ++owner_rank)
    {
        if (owner_rank + 1 >= world_size ||
            token < rank_m_offsets[owner_rank + 1])
        {
            return owner_rank;
        }
    }
    return world_size - 1;
}

__device__ inline int ag_token_chunk_id_ragged(
    int token,
    const int *rank_m_offsets,
    int world_size,
    int signal_chunk_rows,
    int chunks_per_rank)
{
    int owner_rank = 0;
#pragma unroll
    for (int rank = 0; rank < MAX_LOCAL_WORLD_SIZE; ++rank)
    {
        if (rank + 1 >= world_size ||
            token < rank_m_offsets[rank + 1])
        {
            owner_rank = rank;
            break;
        }
    }
    const int local_row = token - rank_m_offsets[owner_rank];
    int chunk = local_row / signal_chunk_rows;
    chunk = chunk < 0 ? 0 : chunk;
    chunk = chunk >= chunks_per_rank ? chunks_per_rank - 1 : chunk;
    return owner_rank * chunks_per_rank + chunk;
}

template <typename IndexT, bool RaggedM>
__global__ void count_topk_by_expert_rank_kernel(const void *topk_ids,
                                                 int *expert_rank_counts,
                                                 int full_m,
                                                 int topk,
                                                 int num_experts,
                                                 int world_size,
                                                 int current_rank,
                                                 bool stage_signal_layout,
                                                 bool zigzag_order,
                                                 const int *rank_m_offsets)
{
    const int total = full_m * topk;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int expert = load_index<IndexT>(topk_ids, idx);
    if (expert >= 0 && expert < num_experts)
    {
        const int token = idx / topk;
        int owner_rank = 0;
        if constexpr (RaggedM)
        {
            owner_rank = ag_owner_rank_from_offsets(
                token, rank_m_offsets, world_size);
        }
        else
        {
            owner_rank = token / (full_m / world_size);
        }
        const int bucket_rank = stage_signal_layout
            ? ag_push_stage_for_owner(
                  owner_rank, current_rank, world_size, zigzag_order)
            : owner_rank;
        atomicAdd(expert_rank_counts + expert * world_size + bucket_rank, 1);
    }
}

__global__ void build_expert_rank_offsets_kernel(
    const int *expert_rank_counts,
    int *expert_rank_offsets,
    int *global_expert_offsets,
    int *rank_cursors,
    int *ntiles,
    int num_experts,
    int world_size,
    int block_m)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    int offset = 0;
    const int rank_stride = world_size + 1;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        global_expert_offsets[expert] = offset;
        int expert_count = 0;
        rank_cursors[expert * rank_stride] = 0;
        for (int rank = 0; rank < world_size; ++rank)
        {
            const int bucket = expert * world_size + rank;
            rank_cursors[expert * rank_stride + rank] = expert_count;
            expert_rank_offsets[bucket] = offset + expert_count;
            expert_count += expert_rank_counts[bucket];
        }
        rank_cursors[expert * rank_stride + world_size] = expert_count;
        offset += ((expert_count + block_m - 1) / block_m) * block_m;
    }
    global_expert_offsets[num_experts] = offset;
    expert_rank_offsets[num_experts * world_size] = offset;
    *ntiles = 0;
}

__global__ void init_sorted_token_ids_kernel(int *sorted_token_ids,
                                             int *sorted_gather_index,
                                             int sorted_capacity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sorted_capacity)
    {
        sorted_token_ids[idx] = -1;
        sorted_gather_index[idx] = 0;
    }
}

template <typename IndexT, bool RaggedM>
__global__ void scatter_sorted_token_ids_by_expert_rank_kernel(
    const void *topk_ids,
    int *sorted_token_ids,
    int *sorted_gather_index,
    int *expert_rank_counters,
    const int *expert_rank_offsets,
    int full_m,
    int topk,
    int num_experts,
    int world_size,
    int current_rank,
    bool stage_signal_layout,
    bool zigzag_order,
    const int *rank_m_offsets)
{
    const int total = full_m * topk;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }
    const int expert = load_index<IndexT>(topk_ids, idx);
    if (expert < 0 || expert >= num_experts)
    {
        return;
    }
    const int token = idx / topk;
    int owner_rank = 0;
    if constexpr (RaggedM)
    {
        owner_rank = ag_owner_rank_from_offsets(
            token, rank_m_offsets, world_size);
    }
    else
    {
        owner_rank = token / (full_m / world_size);
    }
    const int bucket_rank = stage_signal_layout
        ? ag_push_stage_for_owner(
              owner_rank, current_rank, world_size, zigzag_order)
        : owner_rank;
    const int bucket = expert * world_size + bucket_rank;
    const int in_bucket = atomicAdd(expert_rank_counters + bucket, 1);
    const int sorted_pos = expert_rank_offsets[bucket] + in_bucket;
    sorted_token_ids[sorted_pos] = idx;
    if (sorted_gather_index != nullptr)
    {
        sorted_gather_index[sorted_pos] = token;
    }
}

template <typename IndexT, bool RaggedM>
__global__ void scatter_sorted_token_ids_by_expert_rank_stable_kernel(
    const void *topk_ids,
    int *sorted_token_ids,
    int *sorted_gather_index,
    const int *expert_rank_offsets,
    int full_m,
    int topk,
    int num_experts,
    int world_size,
    int current_rank,
    bool stage_signal_layout,
    bool zigzag_order,
    const int *rank_m_offsets)
{
    const int expert = blockIdx.x;
    const int bucket_rank = threadIdx.x;
    if (expert >= num_experts || bucket_rank >= world_size)
    {
        return;
    }

    int owner_rank = bucket_rank;
    if (stage_signal_layout)
    {
        owner_rank = -1;
        for (int candidate = 0; candidate < world_size; ++candidate)
        {
            if (ag_push_stage_for_owner(candidate,
                                        current_rank,
                                        world_size,
                                        zigzag_order) == bucket_rank)
            {
                owner_rank = candidate;
                break;
            }
        }
        if (owner_rank < 0)
        {
            return;
        }
    }

    int token_begin = 0;
    int token_end = 0;
    if constexpr (RaggedM)
    {
        token_begin = rank_m_offsets[owner_rank];
        token_end = rank_m_offsets[owner_rank + 1];
    }
    else
    {
        const int local_m = full_m / world_size;
        token_begin = owner_rank * local_m;
        token_end = owner_rank + 1 == world_size
                        ? full_m
                        : token_begin + local_m;
    }
    const int bucket = expert * world_size + bucket_rank;
    int sorted_pos = expert_rank_offsets[bucket];
    for (int token = token_begin; token < token_end; ++token)
    {
        const int base = token * topk;
        for (int choice = 0; choice < topk; ++choice)
        {
            const int idx = base + choice;
            if (load_index<IndexT>(topk_ids, idx) != expert)
            {
                continue;
            }
            sorted_token_ids[sorted_pos] = idx;
            if (sorted_gather_index != nullptr)
            {
                sorted_gather_index[sorted_pos] = token;
            }
            ++sorted_pos;
        }
    }
}

__device__ inline int find_rank_segment_in_expert(
    const int *rank_cursors,
    int expert,
    int world_size,
    int rel_offset)
{
    int fallback_rank = 0;
    const int rank_stride = world_size + 1;
    for (int rank = 0; rank < MAX_LOCAL_WORLD_SIZE; ++rank)
    {
        if (rank >= world_size)
        {
            break;
        }
        const int start = rank_cursors[expert * rank_stride + rank];
        const int end = rank_cursors[expert * rank_stride + rank + 1];
        if (end > start)
        {
            fallback_rank = rank;
        }
        if (rel_offset < end)
        {
            return rank;
        }
    }
    return fallback_rank;
}

__device__ inline int ag_moe_tile_stage(int segment_end,
                                        int current_rank,
                                        int world_size)
{
    return (segment_end - current_rank + world_size) % world_size;
}

__global__ void build_expert_rank_tile_descriptors_kernel(
    const int *rank_cursors,
    const int *global_expert_offsets,
    int *rank_tile_info,
    int *tile_experts,
    int *tile_starts,
    int *tile_rank_starts,
    int *tile_rank_ends,
    int *ntiles,
    int num_experts,
    int world_size,
    int current_rank,
    bool strict_rank_tile,
    bool split_local_rank,
    bool stage_signal_layout,
    int block_m)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
    {
        return;
    }
    int tile = 0;
    const int rank_stride = world_size + 1;
    for (int expert = 0; expert < num_experts; ++expert)
    {
        int stage_counts[MAX_LOCAL_WORLD_SIZE];
        int stage_offsets[MAX_LOCAL_WORLD_SIZE];
        for (int stage = 0; stage < MAX_LOCAL_WORLD_SIZE; ++stage)
        {
            stage_counts[stage] = 0;
            stage_offsets[stage] = tile;
        }

        const int expert_base = global_expert_offsets[expert];
        const int expert_rows =
            rank_cursors[expert * rank_stride + world_size];

        if (!strict_rank_tile)
        {
            for (int rel_start = 0; rel_start < expert_rows;
                 rel_start += block_m)
            {
                const int remaining = expert_rows - rel_start;
                const int rows = remaining < block_m ? remaining : block_m;
                const int tile_end = rel_start + rows;
                const int tile_rank_start = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, rel_start);
                const int tile_rank_end = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, tile_end - 1);
                const bool split_tile =
                    split_local_rank && tile_rank_start <= current_rank &&
                    current_rank <= tile_rank_end &&
                    tile_rank_start != tile_rank_end;
                if (split_tile)
                {
                    int cursor = rel_start;
                    while (cursor < tile_end)
                    {
                        const int owner_rank = find_rank_segment_in_expert(
                            rank_cursors, expert, world_size, cursor);
                        const int rank_end_offset =
                            rank_cursors[expert * rank_stride + owner_rank + 1];
                        const int segment_end =
                            rank_end_offset > cursor && rank_end_offset < tile_end
                                ? rank_end_offset
                                : tile_end;
                        const int stage = stage_signal_layout
                            ? owner_rank
                            : ag_moe_tile_stage(
                                  owner_rank, current_rank, world_size);
                        stage_counts[stage] += 1;
                        cursor = segment_end;
                    }
                }
                else
                {
                    const int rank_end = find_rank_segment_in_expert(
                        rank_cursors,
                        expert,
                        world_size,
                        rel_start + rows - 1);
                    const int stage = stage_signal_layout
                        ? rank_end
                        : ag_moe_tile_stage(
                              rank_end, current_rank, world_size);
                    stage_counts[stage] += 1;
                }
            }

            int stage_running = tile;
            for (int stage = 0; stage < world_size; ++stage)
            {
                stage_offsets[stage] = stage_running;
                stage_running += stage_counts[stage];
                stage_counts[stage] = 0;
            }

            for (int rel_start = 0; rel_start < expert_rows;
                 rel_start += block_m)
            {
                const int remaining = expert_rows - rel_start;
                const int rows = remaining < block_m ? remaining : block_m;
                const int tile_end = rel_start + rows;
                const int tile_rank_start = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, rel_start);
                const int tile_rank_end = find_rank_segment_in_expert(
                    rank_cursors, expert, world_size, tile_end - 1);
                const bool split_tile =
                    split_local_rank && tile_rank_start <= current_rank &&
                    current_rank <= tile_rank_end &&
                    tile_rank_start != tile_rank_end;
                if (split_tile)
                {
                    int cursor = rel_start;
                    while (cursor < tile_end)
                    {
                        const int owner_rank = find_rank_segment_in_expert(
                            rank_cursors, expert, world_size, cursor);
                        const int rank_end_offset =
                            rank_cursors[expert * rank_stride + owner_rank + 1];
                        const int segment_end =
                            rank_end_offset > cursor && rank_end_offset < tile_end
                                ? rank_end_offset
                                : tile_end;
                        const int piece_rows = segment_end - cursor;
                        const int stage = stage_signal_layout
                            ? owner_rank
                            : ag_moe_tile_stage(
                                  owner_rank, current_rank, world_size);
                        const int out_tile =
                            stage_offsets[stage] + stage_counts[stage]++;
                        int *info = rank_tile_info +
                                    out_tile * AG_RANK_TILE_INFO_STRIDE;
                        info[AG_RANK_TILE_EXPERT] = expert;
                        info[AG_RANK_TILE_SORTED_START] = expert_base + cursor;
                        info[AG_RANK_TILE_ROWS] = piece_rows;
                        info[AG_RANK_TILE_RANK_START] = owner_rank;
                        info[AG_RANK_TILE_RANK_END] = owner_rank;
                        tile_experts[out_tile] = expert;
                        tile_starts[out_tile] = expert_base + cursor;
                        tile_rank_starts[out_tile] = owner_rank;
                        tile_rank_ends[out_tile] = owner_rank;
                        cursor = segment_end;
                    }
                }
                else
                {
                    const int rank_start = find_rank_segment_in_expert(
                        rank_cursors, expert, world_size, rel_start);
                    const int rank_end = find_rank_segment_in_expert(
                        rank_cursors,
                        expert,
                        world_size,
                        rel_start + rows - 1);
                    const int stage = stage_signal_layout
                        ? rank_end
                        : ag_moe_tile_stage(
                              rank_end, current_rank, world_size);
                    const int out_tile =
                        stage_offsets[stage] + stage_counts[stage]++;
                    int *info =
                        rank_tile_info + out_tile * AG_RANK_TILE_INFO_STRIDE;
                    info[AG_RANK_TILE_EXPERT] = expert;
                    info[AG_RANK_TILE_SORTED_START] = expert_base + rel_start;
                    info[AG_RANK_TILE_ROWS] = rows;
                    info[AG_RANK_TILE_RANK_START] = rank_start;
                    info[AG_RANK_TILE_RANK_END] = rank_end;
                    tile_experts[out_tile] = expert;
                    tile_starts[out_tile] = expert_base + rel_start;
                    tile_rank_starts[out_tile] = rank_start;
                    tile_rank_ends[out_tile] = rank_end;
                }
            }
            tile = stage_running;
            continue;
        }

        for (int owner_rank = 0; owner_rank < MAX_LOCAL_WORLD_SIZE; ++owner_rank)
        {
            if (owner_rank >= world_size)
            {
                break;
            }
            const int rank_start =
                rank_cursors[expert * rank_stride + owner_rank];
            const int rank_end =
                rank_cursors[expert * rank_stride + owner_rank + 1];
            const int rank_rows = rank_end - rank_start;
            if (rank_rows <= 0)
            {
                continue;
            }
            const int stage = stage_signal_layout
                ? owner_rank
                : ag_moe_tile_stage(owner_rank, current_rank, world_size);
            stage_counts[stage] += (rank_rows + block_m - 1) / block_m;
        }

        int stage_running = tile;
        for (int stage = 0; stage < world_size; ++stage)
        {
            stage_offsets[stage] = stage_running;
            stage_running += stage_counts[stage];
            stage_counts[stage] = 0;
        }

        for (int owner_rank = 0; owner_rank < MAX_LOCAL_WORLD_SIZE; ++owner_rank)
        {
            if (owner_rank >= world_size)
            {
                break;
            }
            const int rank_start =
                rank_cursors[expert * rank_stride + owner_rank];
            const int rank_end =
                rank_cursors[expert * rank_stride + owner_rank + 1];
            const int rank_rows = rank_end - rank_start;
            if (rank_rows <= 0)
            {
                continue;
            }
            const int stage = stage_signal_layout
                ? owner_rank
                : ag_moe_tile_stage(owner_rank, current_rank, world_size);
            const int tiles_this_rank = (rank_rows + block_m - 1) / block_m;
            for (int t = 0; t < tiles_this_rank; ++t)
            {
                const int rel_start = rank_start + t * block_m;
                const int remaining = rank_rows - t * block_m;
                const int rows =
                    remaining < block_m ? remaining : block_m;
                const int out_tile =
                    stage_offsets[stage] + stage_counts[stage]++;
                int *info =
                    rank_tile_info + out_tile * AG_RANK_TILE_INFO_STRIDE;
                info[AG_RANK_TILE_EXPERT] = expert;
                info[AG_RANK_TILE_SORTED_START] = expert_base + rel_start;
                info[AG_RANK_TILE_ROWS] = rows;
                info[AG_RANK_TILE_RANK_START] = owner_rank;
                info[AG_RANK_TILE_RANK_END] = owner_rank;
                tile_experts[out_tile] = expert;
                tile_starts[out_tile] = expert_base + rel_start;
                tile_rank_starts[out_tile] = owner_rank;
                tile_rank_ends[out_tile] = owner_rank;
            }
        }
        tile = stage_running;
    }
    *ntiles = tile;
}

__global__ void build_rank_tile_problem_info_kernel(
    const int *rank_tile_info,
    const int *sorted_gather_index,
    const int *ntiles,
    int *rank_tile_problem_info,
    int full_m,
    int world_size,
    int signal_chunk_rows,
    int chunks_per_rank,
    int rank_tile_capacity,
    bool nn_direct_chunk_signal,
    const int *rank_m_offsets,
                            bool ragged_m,
                            int current_rank,
                            bool stage_signal_layout)
{
    const int tile = blockIdx.x;
    if (tile >= rank_tile_capacity || tile >= *ntiles)
    {
        return;
    }

    const int *src = rank_tile_info + tile * AG_RANK_TILE_INFO_STRIDE;
    int *dst = rank_tile_problem_info +
               tile * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
    if (threadIdx.x < AG_RANK_TILE_INFO_STRIDE)
    {
        dst[threadIdx.x] = src[threadIdx.x];
    }
    if (threadIdx.x == 0)
    {
        dst[AG_RANK_TILE_PROBLEM_CHUNK_START] = -1;
        dst[AG_RANK_TILE_PROBLEM_CHUNK_END] = -1;
        dst[AG_RANK_TILE_PROBLEM_RESERVED] =
            nn_direct_chunk_signal ? 0x80 : 0;
    }
    __syncthreads();

    // The ragged V40 ring publishes stage slots, not physical owner slots:
    // slot 0 is local and slot s carries owner (current_rank + s) % world.
    // The assembly scheduler treats these two fields as ready-slot indices,
    // so convert a physical owner interval before copying it to contract info.
    // This path is intentionally restricted to ragged ring transport; fixed-M
    // and physical-slot layouts retain the historical ABI.
    if (threadIdx.x == 0 && ragged_m && stage_signal_layout)
    {
        const int owner_start = src[AG_RANK_TILE_RANK_START];
        const int owner_end = src[AG_RANK_TILE_RANK_END];
        if (owner_start >= 0 && owner_end < world_size)
        {
            const int slot_start =
                (owner_start - current_rank + world_size) % world_size;
            int slot_end =
                (owner_end - current_rank + world_size) % world_size;
            if (slot_start > slot_end)
            {
                slot_end = world_size - 1;
            }
            dst[AG_RANK_TILE_RANK_START] = slot_start;
            dst[AG_RANK_TILE_RANK_END] = slot_end;
        }
    }

    if (signal_chunk_rows <= 0 || chunks_per_rank <= 0 ||
        sorted_gather_index == nullptr)
    {
        return;
    }

    const int sorted_start = src[AG_RANK_TILE_SORTED_START];
    const int tile_rows = src[AG_RANK_TILE_ROWS];
    int local_min = world_size * chunks_per_rank;
    int local_max = -1;
    for (int row = threadIdx.x; row < tile_rows; row += blockDim.x)
    {
        const int token = sorted_gather_index[sorted_start + row];
        if (token >= 0 && token < full_m)
        {
            const int chunk_id = ragged_m
                ? ag_token_chunk_id_ragged(token,
                                           rank_m_offsets,
                                           world_size,
                                           signal_chunk_rows,
                                           chunks_per_rank)
                : ag_token_chunk_id(token,
                                    full_m,
                                    world_size,
                                    signal_chunk_rows,
                                    chunks_per_rank);
            local_min = chunk_id < local_min ? chunk_id : local_min;
            local_max = chunk_id > local_max ? chunk_id : local_max;
        }
    }

    __shared__ int chunk_min;
    __shared__ int chunk_max;
    if (threadIdx.x == 0)
    {
        chunk_min = world_size * chunks_per_rank;
        chunk_max = -1;
    }
    __syncthreads();
    atomicMin(&chunk_min, local_min);
    atomicMax(&chunk_max, local_max);
    __syncthreads();
    if (threadIdx.x == 0 && chunk_max >= 0)
    {
        dst[AG_RANK_TILE_PROBLEM_CHUNK_START] = chunk_min;
        dst[AG_RANK_TILE_PROBLEM_CHUNK_END] = chunk_max;
    }
}

template <typename T>
__launch_bounds__(256, 1) __global__
    void ag_group_gemm_tiled_128x256x32_kernel(const T *full_input,
                                               const T *weight,
                                               const int *sorted_token_ids,
                                               const int *tile_experts,
                                               const int *tile_starts,
                                               const int *tile_rank_starts,
                                               const int *tile_rank_ends,
                                               const int *ntiles,
                                               const int *rank_signals,
                                               T *output,
                                               int full_m,
                                               int n_dim,
                                               int k_dim,
                                               int topk,
                                               int world_size,
                                               int signal_value)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];
    const int n_base = tile_n * BN;

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (threadIdx.x == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int linear = threadIdx.x; linear < BM * BN; linear += blockDim.x)
    {
        const int row_in_tile = linear / BN;
        const int n = n_base + (linear - row_in_tile * BN);
        if (n >= n_dim)
        {
            continue;
        }

        const int flat_row = sorted_token_ids[sorted_start + row_in_tile];
        if (flat_row < 0 || flat_row >= full_m * topk)
        {
            continue;
        }

        const int token = flat_row / topk;

        float acc = 0.0f;
        for (int k0 = 0; k0 < k_dim; k0 += BK)
        {
            const int kend = (k0 + BK < k_dim) ? (k0 + BK) : k_dim;
            for (int k = k0; k < kend; ++k)
            {
                const T a = full_input[static_cast<int64_t>(token) * k_dim + k];
                const T b =
                    weight[(static_cast<int64_t>(expert) * k_dim + k) *
                               n_dim +
                           n];
                acc += load_as_float<T>(a) * load_as_float<T>(b);
            }
        }
        output[static_cast<int64_t>(flat_row) * n_dim + n] =
            store_from_float<T>(acc);
    }
}

template <typename T>
__launch_bounds__(256, 1) __global__
    void ag_group_gemm_tiled_128x256x32_smem64x128_kernel(
        const T *full_input,
        const T *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        T *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value)
{
    constexpr int SUB_BM = 64;
    constexpr int SUB_BN = 128;
    constexpr int BK = 32;
    constexpr int COLS_PER_THREAD = 32;

    const int outer_tile_m = blockIdx.x / 2;
    const int row_group = blockIdx.x - outer_tile_m * 2;
    const int n_base = blockIdx.y * SUB_BN;
    if (outer_tile_m >= *ntiles)
    {
        return;
    }

    const int expert = tile_experts[outer_tile_m];
    const int sorted_start = tile_starts[outer_tile_m] + row_group * SUB_BM;
    const int tid = threadIdx.x;
    const int row = tid / 4;
    const int col_base = (tid - row * 4) * COLS_PER_THREAD;

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[outer_tile_m];
        const int segment_end = tile_rank_ends[outer_tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    __shared__ T sh_a[SUB_BM * BK];
    __shared__ T sh_b[BK * SUB_BN];

    float acc[COLS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; ++i)
    {
        acc[i] = 0.0f;
    }

    const int flat_row = sorted_token_ids[sorted_start + row];
    const bool valid_row = flat_row >= 0 && flat_row < full_m * topk;

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        for (int idx = tid; idx < SUB_BM * BK; idx += blockDim.x)
        {
            const int load_row = idx / BK;
            const int load_k = idx - load_row * BK;
            const int load_flat_row = sorted_token_ids[sorted_start + load_row];
            T value = store_from_float<T>(0.0f);
            if (load_flat_row >= 0 && load_flat_row < full_m * topk)
            {
                const int load_token = load_flat_row / topk;
                const int k = k0 + load_k;
                if (k < k_dim)
                {
                    value =
                        full_input[static_cast<int64_t>(load_token) * k_dim + k];
                }
            }
            sh_a[idx] = value;
        }

        for (int idx = tid; idx < BK * SUB_BN; idx += blockDim.x)
        {
            const int load_k = idx / SUB_BN;
            const int load_n = idx - load_k * SUB_BN;
            T value = store_from_float<T>(0.0f);
            const int k = k0 + load_k;
            const int n = n_base + load_n;
            if (k < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        if (valid_row)
        {
#pragma unroll
            for (int kk = 0; kk < BK; ++kk)
            {
                const int k = k0 + kk;
                if (k >= k_dim)
                {
                    break;
                }
                const float a = load_as_float<T>(sh_a[row * BK + kk]);
#pragma unroll
                for (int j = 0; j < COLS_PER_THREAD; ++j)
                {
                    const int col = col_base + j;
                    acc[j] +=
                        a * load_as_float<T>(sh_b[kk * SUB_BN + col]);
                }
            }
        }

        __syncthreads();
    }

    if (valid_row)
    {
#pragma unroll
        for (int j = 0; j < COLS_PER_THREAD; ++j)
        {
            const int n = n_base + col_base + j;
            if (n < n_dim)
            {
                output[static_cast<int64_t>(flat_row) * n_dim + n] =
                    store_from_float<T>(acc[j]);
            }
        }
    }
}

__launch_bounds__(1024, 1) __global__
    void ag_group_gemm_tiled_128x128x16_lds_mmac_f16_kernel(
        const _Float16 *full_input,
        const _Float16 *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        _Float16 *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int WM = 32;
    constexpr int WN = 32;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int WAVES_PER_BLOCK = 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 2;
    const int wave_n = wave_idx & 3;
    const int row_base = wave_m * WM;
    const int col_base = n_base + wave_n * WN;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    __shared__ _Float16 shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    _Float16 *sh_a = shmem;
    _Float16 *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc00;
    ag_float4_t acc01;
    ag_float4_t acc10;
    ag_float4_t acc11;
    zero_vec4(acc00);
    zero_vec4(acc01);
    zero_vec4(acc10);
    zero_vec4(acc11);

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (tid < A_LOADS)
        {
            const int m = tid / (BK / A_LOAD_VEC);
            const int kk = (tid - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int flat_row_a = sorted_token_ids[sorted_start + m];
            const bool valid_a = flat_row_a >= 0 && flat_row_a < full_m * topk;
            const int token_a = valid_a ? (flat_row_a / topk) : 0;
            const int gk = k0 + kk;
            ag_half4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_f16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                sh_a[(kk + i) * BM + m] = value[i];
            }
        }

        for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
        {
            const int kk = idx / BN;
            const int n_off = idx - kk * BN;
            const int n = n_base + n_off;
            _Float16 value = static_cast<_Float16>(0);
            if (k0 + kk < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        const int lane_k = lane / 4;
        const int lane_group = (lane & 3) * 8;
        const uint32_t a_lds_offset0 =
            static_cast<uint32_t>((lane_k * BM + row_base + lane_group) *
                                  sizeof(_Float16));
        const uint32_t b_lds_offset0 =
            static_cast<uint32_t>((LDS_A_ELEMS + lane_k * BN +
                                   wave_n * WN + lane_group) *
                                  sizeof(_Float16));
        const uint32_t a_lds_offset1 =
            static_cast<uint32_t>(((MMAC_K + lane_k) * BM + row_base +
                                   lane_group) *
                                  sizeof(_Float16));
        const uint32_t b_lds_offset1 =
            static_cast<uint32_t>((LDS_A_ELEMS + (MMAC_K + lane_k) * BN +
                                   wave_n * WN + lane_group) *
                                  sizeof(_Float16));

        ag_uint4_t ra0;
        ag_uint4_t rb0;
        ag_uint4_t ra1;
        ag_uint4_t rb1;
        ag_ds_read_m32x16_b16(ra0, a_lds_offset0);
        ag_ds_read_m32x16_b16(rb0, b_lds_offset0);
        ag_ds_read_m32x16_b16(ra1, a_lds_offset1);
        ag_ds_read_m32x16_b16(rb1, b_lds_offset1);
        ag_wait_lgkmcnt0();

        const ag_half4_t a00 = ag_half4_from_u32x2(ra0, 0);
        const ag_half4_t a01 = ag_half4_from_u32x2(ra0, 1);
        const ag_half4_t b00 = ag_half4_from_u32x2(rb0, 0);
        const ag_half4_t b01 = ag_half4_from_u32x2(rb0, 1);
        acc00 = ag_mmac_f16(a00, b00, acc00);
        acc01 = ag_mmac_f16(a00, b01, acc01);
        acc10 = ag_mmac_f16(a01, b00, acc10);
        acc11 = ag_mmac_f16(a01, b01, acc11);

        const ag_half4_t a10 = ag_half4_from_u32x2(ra1, 0);
        const ag_half4_t a11 = ag_half4_from_u32x2(ra1, 1);
        const ag_half4_t b10 = ag_half4_from_u32x2(rb1, 0);
        const ag_half4_t b11 = ag_half4_from_u32x2(rb1, 1);
        acc00 = ag_mmac_f16(a10, b10, acc00);
        acc01 = ag_mmac_f16(a10, b11, acc01);
        acc10 = ag_mmac_f16(a11, b10, acc10);
        acc11 = ag_mmac_f16(a11, b11, acc11);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int col0 = col_base + i * 4 + lane_col4;
        const int row0 = row_base + lane_row16;
        const int row1 = row0 + 16;
        const int flat0 = sorted_token_ids[sorted_start + row0];
        const int flat1 = sorted_token_ids[sorted_start + row1];
        if (flat0 >= 0 && flat0 < full_m * topk && col0 < n_dim)
        {
            output[static_cast<int64_t>(flat0) * n_dim + col0] =
                static_cast<_Float16>(acc00[i]);
        }
        if (flat0 >= 0 && flat0 < full_m * topk && col0 + 16 < n_dim)
        {
            output[static_cast<int64_t>(flat0) * n_dim + col0 + 16] =
                static_cast<_Float16>(acc01[i]);
        }
        if (flat1 >= 0 && flat1 < full_m * topk && col0 < n_dim)
        {
            output[static_cast<int64_t>(flat1) * n_dim + col0] =
                static_cast<_Float16>(acc10[i]);
        }
        if (flat1 >= 0 && flat1 < full_m * topk && col0 + 16 < n_dim)
        {
            output[static_cast<int64_t>(flat1) * n_dim + col0 + 16] =
                static_cast<_Float16>(acc11[i]);
        }
    }
}

__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x128x32_mmac_f16_kernel(
        const _Float16 *full_input,
        const _Float16 *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        _Float16 *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        bool weight_nk,
        int signal_value,
        bool stage_b)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    constexpr int WARP_SIZE = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int NLOOP = BN / 16;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_idx = tid / WARP_SIZE;
    const int lane = tid - warp_idx * WARP_SIZE;
    const int rowid = lane & 15;
    const int col_lane = lane >> 4;
    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];
    const int row_in_tile = warp_idx * 16 + rowid;
    const int flat_row = sorted_token_ids[sorted_start + row_in_tile];
    const bool valid_row = flat_row >= 0 && flat_row < full_m * topk;
    const int token = valid_row ? (flat_row / topk) : 0;
    stage_b = stage_b && !weight_nk;

    __shared__ _Float16 sh_b[BK * BN];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    ag_float4_t acc[NLOOP];
#pragma unroll
    for (int nn = 0; nn < NLOOP; ++nn)
    {
        zero_vec4(acc[nn]);
    }

    const int64_t a_base = static_cast<int64_t>(token) * k_dim;
    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (stage_b)
        {
            for (int idx = tid; idx < BK * BN; idx += blockDim.x)
            {
                const int kk = idx / BN;
                const int n_off = idx - kk * BN;
                const int gk = k0 + kk;
                const int n = n_base + n_off;
                _Float16 value = static_cast<_Float16>(0);
                if (gk < k_dim && n < n_dim)
                {
                    value =
                        weight[(static_cast<int64_t>(expert) * k_dim + gk) *
                                   n_dim +
                               n];
                }
                sh_b[idx] = value;
            }
            __syncthreads();
        }

        const int k_lane = k0 + col_lane * 8;
        ag_half4_t a0;
        ag_half4_t a1;
        zero_vec4(a0);
        zero_vec4(a1);
        if (valid_row)
        {
            a0 = load_f16x4_k_contiguous(full_input, a_base + k_lane, k_lane, k_dim);
            a1 = load_f16x4_k_contiguous(
                full_input, a_base + k_lane + 4, k_lane + 4, k_dim);
        }

#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
            const int b_col = n_base + nn * 16 + rowid;
            ag_half4_t b0;
            ag_half4_t b1;
            if (stage_b)
            {
                const int sh_n = nn * 16 + rowid;
                const int sh_k = col_lane * 8;
                b0 = load_f16x4_weight_shared_bn(sh_b, sh_k, sh_n, BN);
                b1 = load_f16x4_weight_shared_bn(sh_b, sh_k + 4, sh_n, BN);
            }
            else
            {
                b0 = weight_nk ? load_f16x4_weight_nk(weight,
                                                      expert,
                                                      k_lane,
                                                      b_col,
                                                      k_dim,
                                                      n_dim)
                               : load_f16x4_weight_kn(weight,
                                                      expert,
                                                      k_lane,
                                                      b_col,
                                                      k_dim,
                                                      n_dim);
                b1 = weight_nk ? load_f16x4_weight_nk(weight,
                                                      expert,
                                                      k_lane + 4,
                                                      b_col,
                                                      k_dim,
                                                      n_dim)
                               : load_f16x4_weight_kn(weight,
                                                      expert,
                                                      k_lane + 4,
                                                      b_col,
                                                      k_dim,
                                                      n_dim);
            }
            acc[nn] = ag_mmac_f16(a0, b0, acc[nn]);
            acc[nn] = ag_mmac_f16(a1, b1, acc[nn]);
        }
        if (stage_b)
        {
            __syncthreads();
        }
    }

    if (valid_row)
    {
#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = n_base + nn * 16 + col_lane + i * 4;
                if (n < n_dim)
                {
                    output[static_cast<int64_t>(flat_row) * n_dim + n] =
                        static_cast<_Float16>(acc[nn][i]);
                }
            }
        }
    }
}

__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x128x32_mmac_bf16_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
        const int *sorted_token_ids,
        const int *tile_experts,
        const int *tile_starts,
        const int *tile_rank_starts,
        const int *tile_rank_ends,
        const int *ntiles,
        const int *rank_signals,
        uint16_t *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        bool weight_nk,
        int signal_value)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int WARP_SIZE = 64;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int NLOOP = BN / 16;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_idx = tid / WARP_SIZE;
    const int lane = tid - warp_idx * WARP_SIZE;
    const int rowid = lane & 15;
    const int col_lane = lane >> 4;
    const int expert = tile_experts[tile_m];
    const int sorted_start = tile_starts[tile_m];
    const int row_in_tile = warp_idx * 16 + rowid;
    const int flat_row = sorted_token_ids[sorted_start + row_in_tile];
    const bool valid_row = flat_row >= 0 && flat_row < full_m * topk;
    const int token = valid_row ? (flat_row / topk) : 0;

    if (rank_signals != nullptr)
    {
        const int segment_start = tile_rank_starts[tile_m];
        const int segment_end = tile_rank_ends[tile_m];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (tid == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    const volatile int *signal =
                        reinterpret_cast<const volatile int *>(rank_signals +
                                                               src_rank);
                    while (*signal < signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    ag_float4_t acc[NLOOP];
#pragma unroll
    for (int nn = 0; nn < NLOOP; ++nn)
    {
        zero_vec4(acc[nn]);
    }

    const int64_t a_base = static_cast<int64_t>(token) * k_dim;
    for (int k0 = 0; k0 < k_dim; k0 += 32)
    {
        const int k_lane = k0 + col_lane * 8;
        ag_bf16x4_t a0;
        ag_bf16x4_t a1;
        zero_vec4(a0);
        zero_vec4(a1);
        if (valid_row)
        {
            a0 = load_bf16x4_k_contiguous(
                full_input, a_base + k_lane, k_lane, k_dim);
            a1 = load_bf16x4_k_contiguous(
                full_input, a_base + k_lane + 4, k_lane + 4, k_dim);
        }

#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
            const int b_col = n_base + nn * 16 + rowid;
            ag_bf16x4_t b0 =
                weight_nk ? load_bf16x4_weight_nk(
                                weight, expert, k_lane, b_col, k_dim, n_dim)
                          : load_bf16x4_weight_kn(
                                weight, expert, k_lane, b_col, k_dim, n_dim);
            ag_bf16x4_t b1 =
                weight_nk ? load_bf16x4_weight_nk(weight,
                                                  expert,
                                                  k_lane + 4,
                                                  b_col,
                                                  k_dim,
                                                  n_dim)
                          : load_bf16x4_weight_kn(weight,
                                                  expert,
                                                  k_lane + 4,
                                                  b_col,
                                                  k_dim,
                                                  n_dim);
            acc[nn] = ag_mmac_bf16(a0, b0, acc[nn]);
            acc[nn] = ag_mmac_bf16(a1, b1, acc[nn]);
        }
    }

    if (valid_row)
    {
#pragma unroll
        for (int nn = 0; nn < NLOOP; ++nn)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = n_base + nn * 16 + col_lane + i * 4;
                if (n < n_dim)
                {
                    output[static_cast<int64_t>(flat_row) * n_dim + n] =
                        ag_float_to_bf16_bits(acc[nn][i]);
                }
            }
        }
    }
}

template <bool UseTileReady>
__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_m32_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
	        const int *sorted_token_ids,
	        const int *sorted_gather_index,
	        const int *rank_tile_info,
	        const int *rank_tile_problem_info,
	        const int *ntiles,
	        const int *rank_signals,
	        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
	        int k_dim,
	        int topk,
	        int world_size,
	        int signal_value,
	        int signal_chunk_rows,
	        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int N_CHUNKS = 2;
    constexpr int WAVE_M = 32;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 1;
    const int wave_n_chunk = wave_idx & 1;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

	    const int *tile_info =
	        rank_tile_problem_info != nullptr
	            ? rank_tile_problem_info +
	                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
	            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
	    const int expert = tile_info[AG_RANK_TILE_EXPERT];
	    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
	    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

	    if (rank_signals != nullptr)
	    {
	        const bool use_chunk_signal =
	            signal_chunk_rows > 0 && chunks_per_rank > 0;
	        if (use_chunk_signal)
	        {
	            if (rank_tile_problem_info != nullptr)
	            {
	                const int chunk_start =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
	                const int chunk_end =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
	                if (chunk_start >= 0 && chunk_end >= chunk_start)
	                {
	                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
	                         chunk += blockDim.x)
	                    {
	                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
	                    }
	                }
	            }
	            else
	            {
	                for (int row = tid; row < tile_rows; row += blockDim.x)
	                {
	                    const int token =
	                        sorted_gather_index[sorted_start + row];
	                    if (token >= 0 && token < full_m)
	                    {
	                        const int chunk_id =
	                            ag_token_chunk_id(token,
	                                              full_m,
	                                              world_size,
	                                              signal_chunk_rows,
	                                              chunks_per_rank);
	                        ag_wait_signal_ge(rank_signals,
	                                          chunk_id,
	                                          signal_value);
	                    }
	                }
	            }
	            __syncthreads();
	        }
	        else
	        {
	            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
	            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
	            if (segment_start >= 0 && segment_end < world_size)
	            {
	                if constexpr (UseTileReady)
	                {
	                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
	                    {
	                        if (tid == 0)
	                        {
	                            ag_wait_signal_ge(rank_tile_ready_flags,
	                                              tile_m,
	                                              signal_value);
	                        }
	                    }
	                    else if (tid == 0)
	                    {
	                        for (int src_rank = segment_start;
	                             src_rank <= segment_end;
	                             ++src_rank)
	                        {
	                            ag_wait_signal_ge(rank_signals,
	                                              src_rank,
	                                              signal_value);
	                        }
	                        if (rank_tile_ready_flags != nullptr)
	                        {
	                            __threadfence_system();
	                            atomicExch(rank_tile_ready_flags + tile_m,
	                                       signal_value);
	                        }
	                    }
	                }
	                else if (tid == 0)
	                {
	                    for (int src_rank = segment_start;
	                         src_rank <= segment_end;
	                         ++src_rank)
	                    {
	                        ag_wait_signal_ge(rank_signals,
	                                          src_rank,
	                                          signal_value);
	                    }
	                }
	                __syncthreads();
	            }
	        }
	    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc0[N_MMACS_PER_WAVE];
    ag_float4_t acc1[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc0[nn]);
        zero_vec4(acc1[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        for (int load = tid; load < A_LOADS; load += blockDim.x)
        {
            const int m = load / (BK / A_LOAD_VEC);
            const int kk = (load - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                sh_a[(kk + i) * BM + m] = value[i];
            }
        }

        for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
        {
            const int kk = idx / BN;
            const int n_off = idx - kk * BN;
            const int n = n_base + n_off;
            uint16_t value = 0;
            if (k0 + kk < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        const int lane_k = lane / 4;
        const int lane_group = (lane & 3) * 8;
#pragma unroll
        for (int kk0 = 0; kk0 < BK; kk0 += MMAC_K)
        {
            const uint32_t a_lds_offset =
                static_cast<uint32_t>(((kk0 + lane_k) * BM + row_base +
                                       lane_group) *
                                      sizeof(uint16_t));
            ag_uint4_t ra;
            ag_ds_read_m32x16_b16(ra, a_lds_offset);

#pragma unroll
            for (int nn32 = 0; nn32 < N_MMACS_PER_WAVE / 2; ++nn32)
            {
                const uint32_t b_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           (kk0 + lane_k) * BN +
                                           wave_n_chunk * N_PER_WAVE +
                                           nn32 * 32 + lane_group) *
                                          sizeof(uint16_t));
                ag_uint4_t rb;
                ag_ds_read_m32x16_b16(rb, b_lds_offset);
                ag_wait_lgkmcnt0();

                const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);
                const ag_bf16x4_t b0 = ag_bf16x4_from_u32x2(rb, 0);
                const ag_bf16x4_t b1 = ag_bf16x4_from_u32x2(rb, 1);
                acc0[nn32 * 2 + 0] =
                    ag_mmac_bf16(a0, b0, acc0[nn32 * 2 + 0]);
                acc0[nn32 * 2 + 1] =
                    ag_mmac_bf16(a0, b1, acc0[nn32 * 2 + 1]);
                acc1[nn32 * 2 + 0] =
                    ag_mmac_bf16(a1, b0, acc1[nn32 * 2 + 0]);
                acc1[nn32 * 2 + 1] =
                    ag_mmac_bf16(a1, b1, acc1[nn32 * 2 + 1]);
            }
        }
        __syncthreads();
    }

    const int row0 = row_base + lane_row16;
    const int row1 = row0 + 16;
    const int flat0 =
        (row0 < tile_rows) ? sorted_token_ids[sorted_start + row0] : -1;
    const int flat1 =
        (row1 < tile_rows) ? sorted_token_ids[sorted_start + row1] : -1;
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = col_base + nn * 16 + lane_col4 + i * 4;
            if (flat0 >= 0 && flat0 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat0) * n_dim + n] =
                    ag_float_to_bf16_bits(acc0[nn][i]);
            }
            if (flat1 >= 0 && flat1 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat1) * n_dim + n] =
                    ag_float_to_bf16_bits(acc1[nn][i]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

// V6 handles partial MT256 expert tails with four waves. The packed NK
// weight layout is shared with the WGM8 assembly main path.
template <int LOAD_VEC>
__launch_bounds__(256, 1) __global__
    void ag_group_gemm_tail_64x256x32_lds_mmac_bf16_nk_load_vec_kernel(
        const uint16_t *__restrict__ full_input,
        const uint16_t *__restrict__ weight_nk,
        const int *__restrict__ sorted_token_ids,
        const int *__restrict__ sorted_gather_index,
        const int *__restrict__ tail_tile_info,
        const int *__restrict__ tail_count,
        const int *__restrict__ rank_signals,
        uint16_t *__restrict__ output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value)
{
    static_assert(LOAD_VEC == 4 || LOAD_VEC == 8,
                  "V6 tail load width must be 4 or 8 BF16 values");
    constexpr int BM = 64;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int N_PER_WAVE = 128;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = LOAD_VEC;
    constexpr int B_LOAD_VEC = LOAD_VEC;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int B_LOADS = BN * (BK / B_LOAD_VEC);
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;
    using LoadType = typename std::conditional<
        LOAD_VEC == 8, ag_bf16x8_t, ag_bf16x4_t>::type;

    const int tile_m = blockIdx.y;
    if (tile_m >= *tail_count)
    {
        return;
    }
    const int n_base = blockIdx.x * BN;
    const int *tile =
        tail_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
    const int expert = tile[AG_RANK_TILE_EXPERT];
    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int tile_rows = tile[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr && signal_value > 0)
    {
        const int segment_start = tile[AG_RANK_TILE_RANK_START];
        const int segment_end = tile[AG_RANK_TILE_RANK_END];
        if (segment_start >= 0 && segment_end < world_size)
        {
            if (threadIdx.x == 0)
            {
                for (int src_rank = segment_start; src_rank <= segment_end;
                     ++src_rank)
                {
                    while (__hip_atomic_load(rank_signals + src_rank,
                                             __ATOMIC_ACQUIRE,
                                             __HIP_MEMORY_SCOPE_SYSTEM) <
                           signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
            }
            __syncthreads();
        }
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 1;
    const int wave_n = wave_idx & 1;
    const int row_base = wave_m * 32;
    const int col_base = n_base + wave_n * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc0[N_MMACS_PER_WAVE];
    ag_float4_t acc1[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc0[nn]);
        zero_vec4(acc1[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        for (int load = tid; load < A_LOADS; load += blockDim.x)
        {
            const int m = load / (BK / A_LOAD_VEC);
            const int kk = (load - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token =
                m < tile_rows ? sorted_gather_index[sorted_start + m] : -1;
            LoadType value;
#pragma unroll
            for (int i = 0; i < LOAD_VEC; ++i)
            {
                value[i] = 0;
            }
            if (token >= 0 && token < full_m)
            {
                value = *reinterpret_cast<const LoadType *>(
                    full_input + static_cast<int64_t>(token) * k_dim + k0 + kk);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                sh_a[(kk + i) * BM + m] = value[i];
            }
        }

        for (int load = tid; load < B_LOADS; load += blockDim.x)
        {
            const int n_off = load / (BK / B_LOAD_VEC);
            const int kk = (load - n_off * (BK / B_LOAD_VEC)) * B_LOAD_VEC;
            const int n = n_base + n_off;
            LoadType value;
#pragma unroll
            for (int i = 0; i < LOAD_VEC; ++i)
            {
                value[i] = 0;
            }
            if (n < n_dim)
            {
                const int64_t base =
                    (static_cast<int64_t>(expert) * n_dim + n) * k_dim +
                    k0 + kk;
                value = *reinterpret_cast<const LoadType *>(weight_nk + base);
            }
#pragma unroll
            for (int i = 0; i < B_LOAD_VEC; ++i)
            {
                sh_b[(kk + i) * BN + n_off] = value[i];
            }
        }

        __syncthreads();
        const int lane_k = lane / 4;
        const int lane_group = (lane & 3) * 8;
#pragma unroll
        for (int kk0 = 0; kk0 < BK; kk0 += MMAC_K)
        {
            const uint32_t a_lds_offset = static_cast<uint32_t>(
                ((kk0 + lane_k) * BM + row_base + lane_group) *
                sizeof(uint16_t));
            ag_uint4_t ra;
            ag_ds_read_m32x16_b16(ra, a_lds_offset);

#pragma unroll
            for (int nn32 = 0; nn32 < N_MMACS_PER_WAVE / 2; ++nn32)
            {
                const uint32_t b_lds_offset = static_cast<uint32_t>(
                    (LDS_A_ELEMS + (kk0 + lane_k) * BN +
                     wave_n * N_PER_WAVE + nn32 * 32 + lane_group) *
                    sizeof(uint16_t));
                ag_uint4_t rb;
                ag_ds_read_m32x16_b16(rb, b_lds_offset);
                ag_wait_lgkmcnt0();

                const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);
                const ag_bf16x4_t b0 = ag_bf16x4_from_u32x2(rb, 0);
                const ag_bf16x4_t b1 = ag_bf16x4_from_u32x2(rb, 1);
                acc0[nn32 * 2] =
                    ag_mmac_bf16(a0, b0, acc0[nn32 * 2]);
                acc0[nn32 * 2 + 1] =
                    ag_mmac_bf16(a0, b1, acc0[nn32 * 2 + 1]);
                acc1[nn32 * 2] =
                    ag_mmac_bf16(a1, b0, acc1[nn32 * 2]);
                acc1[nn32 * 2 + 1] =
                    ag_mmac_bf16(a1, b1, acc1[nn32 * 2 + 1]);
            }
        }
        __syncthreads();
    }

    const int row0 = row_base + lane_row16;
    const int row1 = row0 + 16;
    const int flat0 =
        row0 < tile_rows ? sorted_token_ids[sorted_start + row0] : -1;
    const int flat1 =
        row1 < tile_rows ? sorted_token_ids[sorted_start + row1] : -1;
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = col_base + nn * 16 + lane_col4 + i * 4;
            if (flat0 >= 0 && flat0 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat0) * n_dim + n] =
                    ag_float_to_bf16_bits(acc0[nn][i]);
            }
            if (flat1 >= 0 && flat1 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat1) * n_dim + n] =
                    ag_float_to_bf16_bits(acc1[nn][i]);
            }
        }
    }
}

template <bool UseTileReady, bool PairBReads>
__launch_bounds__(1024, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
	        const int *sorted_token_ids,
	        const int *sorted_gather_index,
	        const int *rank_tile_info,
	        const int *rank_tile_problem_info,
	        const int *ntiles,
	        const int *rank_signals,
	        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
	        int k_dim,
	        int topk,
	        int world_size,
	        int signal_value,
	        int signal_chunk_rows,
	        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int N_CHUNKS = 4;
    constexpr int WAVE_M = 32;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx / N_CHUNKS;
    const int wave_n_chunk = wave_idx - wave_m * N_CHUNKS;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

	    const int *tile_info =
	        rank_tile_problem_info != nullptr
	            ? rank_tile_problem_info +
	                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
	            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
	    const int expert = tile_info[AG_RANK_TILE_EXPERT];
	    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
	    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

	    if (rank_signals != nullptr)
	    {
	        const bool use_chunk_signal =
	            signal_chunk_rows > 0 && chunks_per_rank > 0;
	        if (use_chunk_signal)
	        {
	            if (rank_tile_problem_info != nullptr)
	            {
	                const int chunk_start =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
	                const int chunk_end =
	                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
	                if (chunk_start >= 0 && chunk_end >= chunk_start)
	                {
	                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
	                         chunk += blockDim.x)
	                    {
	                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
	                    }
	                }
	            }
	            else
	            {
	                for (int row = tid; row < tile_rows; row += blockDim.x)
	                {
	                    const int token =
	                        sorted_gather_index[sorted_start + row];
	                    if (token >= 0 && token < full_m)
	                    {
	                        const int chunk_id =
	                            ag_token_chunk_id(token,
	                                              full_m,
	                                              world_size,
	                                              signal_chunk_rows,
	                                              chunks_per_rank);
	                        ag_wait_signal_ge(rank_signals,
	                                          chunk_id,
	                                          signal_value);
	                    }
	                }
	            }
	            __syncthreads();
	        }
	        else
	        {
	            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
	            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
	            if (segment_start >= 0 && segment_end < world_size)
	            {
	                if constexpr (UseTileReady)
	                {
	                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
	                    {
	                        if (tid == 0)
	                        {
	                            ag_wait_signal_ge(rank_tile_ready_flags,
	                                              tile_m,
	                                              signal_value);
	                        }
	                    }
	                    else if (tid == 0)
	                    {
	                        for (int src_rank = segment_start;
	                             src_rank <= segment_end;
	                             ++src_rank)
	                        {
	                            ag_wait_signal_ge(rank_signals,
	                                              src_rank,
	                                              signal_value);
	                        }
	                        if (rank_tile_ready_flags != nullptr)
	                        {
	                            __threadfence_system();
	                            atomicExch(rank_tile_ready_flags + tile_m,
	                                       signal_value);
	                        }
	                    }
	                }
	                else if (tid == 0)
	                {
	                    for (int src_rank = segment_start;
	                         src_rank <= segment_end;
	                         ++src_rank)
	                    {
	                        ag_wait_signal_ge(rank_signals,
	                                          src_rank,
	                                          signal_value);
	                    }
	                }
	                __syncthreads();
	            }
	        }
	    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc0[N_MMACS_PER_WAVE];
    ag_float4_t acc1[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc0[nn]);
        zero_vec4(acc1[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (tid < A_LOADS)
        {
            const int m = tid / (BK / A_LOAD_VEC);
            const int kk = (tid - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                sh_a[(kk + i) * BM + m] = value[i];
            }
        }

        for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
        {
            const int kk = idx / BN;
            const int n_off = idx - kk * BN;
            const int n = n_base + n_off;
            uint16_t value = 0;
            if (k0 + kk < k_dim && n < n_dim)
            {
                value = weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
            }
            sh_b[idx] = value;
        }

        __syncthreads();

        const int lane_k = lane / 4;
        const int lane_group = (lane & 3) * 8;
#pragma unroll
        for (int kk0 = 0; kk0 < BK; kk0 += MMAC_K)
        {
            const uint32_t a_lds_offset =
                static_cast<uint32_t>(((kk0 + lane_k) * BM + row_base +
                                       lane_group) *
                                      sizeof(uint16_t));
            ag_uint4_t ra;
            ag_ds_read_m32x16_b16(ra, a_lds_offset);

            if constexpr (PairBReads)
            {
                ag_uint4_t rb0;
                ag_uint4_t rb1;
                const uint32_t b0_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           (kk0 + lane_k) * BN +
                                           wave_n_chunk * N_PER_WAVE +
                                           lane_group) *
                                          sizeof(uint16_t));
                const uint32_t b1_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           (kk0 + lane_k) * BN +
                                           wave_n_chunk * N_PER_WAVE + 32 +
                                           lane_group) *
                                          sizeof(uint16_t));
                ag_ds_read_m32x16_b16(rb0, b0_lds_offset);
                ag_ds_read_m32x16_b16(rb1, b1_lds_offset);
                ag_wait_lgkmcnt0();

                const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);
                const ag_bf16x4_t b00 = ag_bf16x4_from_u32x2(rb0, 0);
                const ag_bf16x4_t b01 = ag_bf16x4_from_u32x2(rb0, 1);
                const ag_bf16x4_t b10 = ag_bf16x4_from_u32x2(rb1, 0);
                const ag_bf16x4_t b11 = ag_bf16x4_from_u32x2(rb1, 1);
                acc0[0] = ag_mmac_bf16(a0, b00, acc0[0]);
                acc0[1] = ag_mmac_bf16(a0, b01, acc0[1]);
                acc1[0] = ag_mmac_bf16(a1, b00, acc1[0]);
                acc1[1] = ag_mmac_bf16(a1, b01, acc1[1]);
                acc0[2] = ag_mmac_bf16(a0, b10, acc0[2]);
                acc0[3] = ag_mmac_bf16(a0, b11, acc0[3]);
                acc1[2] = ag_mmac_bf16(a1, b10, acc1[2]);
                acc1[3] = ag_mmac_bf16(a1, b11, acc1[3]);
            }
            else
            {
#pragma unroll
                for (int nn32 = 0; nn32 < N_MMACS_PER_WAVE / 2; ++nn32)
                {
                    const uint32_t b_lds_offset =
                        static_cast<uint32_t>((LDS_A_ELEMS +
                                               (kk0 + lane_k) * BN +
                                               wave_n_chunk * N_PER_WAVE +
                                               nn32 * 32 + lane_group) *
                                              sizeof(uint16_t));
                    ag_uint4_t rb;
                    ag_ds_read_m32x16_b16(rb, b_lds_offset);
                    ag_wait_lgkmcnt0();

                    const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                    const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);
                    const ag_bf16x4_t b0 = ag_bf16x4_from_u32x2(rb, 0);
                    const ag_bf16x4_t b1 = ag_bf16x4_from_u32x2(rb, 1);
                    acc0[nn32 * 2 + 0] =
                        ag_mmac_bf16(a0, b0, acc0[nn32 * 2 + 0]);
                    acc0[nn32 * 2 + 1] =
                        ag_mmac_bf16(a0, b1, acc0[nn32 * 2 + 1]);
                    acc1[nn32 * 2 + 0] =
                        ag_mmac_bf16(a1, b0, acc1[nn32 * 2 + 0]);
                    acc1[nn32 * 2 + 1] =
                        ag_mmac_bf16(a1, b1, acc1[nn32 * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }

    const int row0 = row_base + lane_row16;
    const int row1 = row0 + 16;
    const int flat0 =
        (row0 < tile_rows) ? sorted_token_ids[sorted_start + row0] : -1;
    const int flat1 =
        (row1 < tile_rows) ? sorted_token_ids[sorted_start + row1] : -1;
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = col_base + nn * 16 + lane_col4 + i * 4;
            if (flat0 >= 0 && flat0 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat0) * n_dim + n] =
                    ag_float_to_bf16_bits(acc0[nn][i]);
            }
            if (flat1 >= 0 && flat1 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat1) * n_dim + n] =
                    ag_float_to_bf16_bits(acc1[nn][i]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

// Build one expert-contiguous copy of both wgrad operands.  The former wgrad
// kernel repeated sorted-token and sorted-gather indirections in every
// (expert, K tile, N tile) CTA.  For the production FC1 shape that meant the
// same routed row was gathered hundreds of times.  Keep the routing lookup in
// this bandwidth-only pass so the MMAC kernel below sees two regular matrices:
//
//   compact_x [expert_base + m, K]
//   compact_dy[expert_base + m, N]
//
// Expert bases retain the metadata builder's 256-row alignment.  Padding is
// never read because rank_cursors carries the exact M_e for every expert.
__launch_bounds__(256, 1) __global__
    void ag_group_gemm_bf16_wgrad_compact_inputs_kernel(
        const uint16_t *routed_input,
        const uint16_t *grad_output,
        const int *sorted_token_ids,
        const int *sorted_gather_index,
        uint16_t *compact_x,
        uint16_t *compact_dy,
        int sorted_capacity,
        int full_m,
        int topk,
        int x_dim,
        int dy_dim)
{
    constexpr int VEC = 8;
    const int sorted_row = blockIdx.x;
    if (sorted_row >= sorted_capacity)
    {
        return;
    }

    const int flat_row = sorted_token_ids[sorted_row];
    if (flat_row < 0 || flat_row >= full_m * topk)
    {
        return;
    }
    const int token = sorted_gather_index[sorted_row];
    if (token < 0 || token >= full_m)
    {
        return;
    }

    for (int col = threadIdx.x * VEC; col < x_dim;
         col += blockDim.x * VEC)
    {
        const int64_t src = static_cast<int64_t>(flat_row) * x_dim + col;
        const int64_t dst = static_cast<int64_t>(sorted_row) * x_dim + col;
        if (col + VEC <= x_dim)
        {
            *reinterpret_cast<ag_bf16x8_t *>(compact_x + dst) =
                *reinterpret_cast<const ag_bf16x8_t *>(routed_input + src);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < VEC; ++i)
            {
                if (col + i < x_dim)
                {
                    compact_x[dst + i] = routed_input[src + i];
                }
            }
        }
    }

    for (int col = threadIdx.x * VEC; col < dy_dim;
         col += blockDim.x * VEC)
    {
        const int64_t src = static_cast<int64_t>(token) * dy_dim + col;
        const int64_t dst = static_cast<int64_t>(sorted_row) * dy_dim + col;
        if (col + VEC <= dy_dim)
        {
            *reinterpret_cast<ag_bf16x8_t *>(compact_dy + dst) =
                *reinterpret_cast<const ag_bf16x8_t *>(grad_output + src);
        }
        else
        {
#pragma unroll
            for (int i = 0; i < VEC; ++i)
            {
                if (col + i < dy_dim)
                {
                    compact_dy[dst + i] = grad_output[src + i];
                }
            }
        }
    }
}

constexpr int AG_WGRAD_GROUPED_ASM_ARGS_BYTES = 196;

__device__ inline void ag_wgrad_grouped_store_u32(uint8_t *base,
                                                   int offset,
                                                   uint32_t value)
{
    *reinterpret_cast<uint32_t *>(base + offset) = value;
}

__device__ inline void ag_wgrad_grouped_store_ptr(uint8_t *base,
                                                   int offset,
                                                   const void *ptr)
{
    const uintptr_t value = reinterpret_cast<uintptr_t>(ptr);
    ag_wgrad_grouped_store_u32(
        base, offset, static_cast<uint32_t>(value));
    ag_wgrad_grouped_store_u32(
        base, offset + 4, static_cast<uint32_t>(value >> 32));
}

__device__ inline void ag_wgrad_grouped_store_f32(uint8_t *base,
                                                   int offset,
                                                   float value)
{
    union
    {
        float f;
        uint32_t u;
    } bits{value};
    ag_wgrad_grouped_store_u32(base, offset, bits.u);
}

// Build the 196-byte Tensile grouped-GEMM descriptor directly on the device.
// One problem represents one expert:
//
//   C[i,j] = sum_l A[i,l] * B[j,l]
//   A = compact_dy[M_e,N], B = compact_x[M_e,K]
//   SizeI=N, SizeJ=K, SizeL=M_e
//
// The assembly output is column-major C[N,K], whose physical storage is the
// row-major FC1 dW[K,N].  Therefore no explicit input or output transpose is
// needed.  output_k_start/extent let the existing hidden-dimension ready
// signals launch the same kernel a chunk at a time.
__launch_bounds__(64, 1) __global__
    void ag_group_gemm_bf16_wgrad_build_grouped_asm_args_kernel(
        const uint16_t *compact_x,
        const uint16_t *compact_dy,
        const int *global_expert_offsets,
        const int *rank_cursors,
        uint16_t *grad_weight,
        uint8_t *grouped_asm_args,
        int output_dim,
        int reduction_dim,
        int num_experts,
        int world_size,
        int output_k_start,
        int output_k_extent)
{
    const int expert = blockIdx.x;
    if (expert >= num_experts)
    {
        return;
    }
    uint8_t *args = grouped_asm_args +
        static_cast<int64_t>(expert) * AG_WGRAD_GROUPED_ASM_ARGS_BYTES;
    for (int word = threadIdx.x;
         word < AG_WGRAD_GROUPED_ASM_ARGS_BYTES / 4;
         word += blockDim.x)
    {
        reinterpret_cast<uint32_t *>(args)[word] = 0;
    }
    __syncthreads();
    if (threadIdx.x != 0)
    {
        return;
    }

    const int expert_base = global_expert_offsets[expert];
    const int expert_rows =
        rank_cursors[expert * (world_size + 1) + world_size];
    const uint16_t *problem_a =
        compact_dy + static_cast<int64_t>(expert_base) * reduction_dim;
    const uint16_t *problem_b =
        compact_x + static_cast<int64_t>(expert_base) * output_dim +
        output_k_start;
    uint16_t *problem_d =
        grad_weight +
        (static_cast<int64_t>(expert) * output_dim + output_k_start) *
            reduction_dim;

    ag_wgrad_grouped_store_u32(
        args, 0x00, static_cast<uint32_t>(reduction_dim)); // SizeI = N
    ag_wgrad_grouped_store_u32(
        args, 0x04, static_cast<uint32_t>(output_k_extent)); // SizeJ = K
    ag_wgrad_grouped_store_u32(args, 0x08, 1u); // SizeK / batch
    ag_wgrad_grouped_store_u32(
        args, 0x0c, static_cast<uint32_t>(expert_rows)); // SizeL = M_e
    ag_wgrad_grouped_store_ptr(args, 0x10, problem_d); // AddressD
    ag_wgrad_grouped_store_ptr(args, 0x18, problem_d); // AddressC, beta=0
    ag_wgrad_grouped_store_ptr(args, 0x20, problem_a); // AddressA = dY
    ag_wgrad_grouped_store_ptr(args, 0x28, problem_b); // AddressB = X
    ag_wgrad_grouped_store_u32(
        args, 0x30, static_cast<uint32_t>(reduction_dim));
    ag_wgrad_grouped_store_u32(
        args,
        0x34,
        static_cast<uint32_t>(
            static_cast<int64_t>(output_dim) * reduction_dim));
    ag_wgrad_grouped_store_u32(
        args, 0x38, static_cast<uint32_t>(reduction_dim));
    ag_wgrad_grouped_store_u32(
        args,
        0x3c,
        static_cast<uint32_t>(
            static_cast<int64_t>(output_dim) * reduction_dim));
    ag_wgrad_grouped_store_u32(
        args, 0x40, static_cast<uint32_t>(reduction_dim)); // StrideAL
    ag_wgrad_grouped_store_u32(
        args,
        0x44,
        static_cast<uint32_t>(
            static_cast<int64_t>(max(expert_rows, 1)) * reduction_dim));
    ag_wgrad_grouped_store_u32(
        args, 0x48, static_cast<uint32_t>(output_dim)); // StrideBL
    ag_wgrad_grouped_store_u32(
        args,
        0x4c,
        static_cast<uint32_t>(
            static_cast<int64_t>(max(expert_rows, 1)) * output_dim));
    ag_wgrad_grouped_store_f32(args, 0x50, 1.0f); // Alpha
    ag_wgrad_grouped_store_f32(args, 0x60, 0.0f); // Beta
}

template <bool UseTileReady, bool UseDsReadSwizzle = false>
__launch_bounds__(512, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
        const int *sorted_token_ids,
        const int *sorted_gather_index,
        const int *rank_tile_info,
        const int *rank_tile_problem_info,
        const int *ntiles,
        const int *rank_signals,
        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value,
        int signal_chunk_rows,
        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int WARP_SIZE = 64;
    constexpr int N_CHUNKS = 2;
    constexpr int WAVE_M = 32;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int B_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int B_LOADS_VEC = BK * (BN / B_LOAD_VEC);
    constexpr int FRAGMENT_ELEMS = 16 * BK;
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 1;
    const int wave_n_chunk = wave_idx & 1;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

    const int *tile_info =
        rank_tile_problem_info != nullptr
            ? rank_tile_problem_info +
                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
    const int expert = tile_info[AG_RANK_TILE_EXPERT];
    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const bool use_chunk_signal =
            signal_chunk_rows > 0 && chunks_per_rank > 0;
        if (use_chunk_signal)
        {
            if (rank_tile_problem_info != nullptr)
            {
                const int chunk_start =
                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
                const int chunk_end = tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
                if (chunk_start >= 0 && chunk_end >= chunk_start)
                {
                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
                         chunk += blockDim.x)
                    {
                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
                    }
                }
            }
            else
            {
                for (int row = tid; row < tile_rows; row += blockDim.x)
                {
                    const int token = sorted_gather_index[sorted_start + row];
                    if (token >= 0 && token < full_m)
                    {
                        const int chunk_id =
                            ag_token_chunk_id(token,
                                              full_m,
                                              world_size,
                                              signal_chunk_rows,
                                              chunks_per_rank);
                        ag_wait_signal_ge(rank_signals, chunk_id, signal_value);
                    }
                }
            }
            __syncthreads();
        }
        else
        {
            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
            if (segment_start >= 0 && segment_end < world_size)
            {
                if constexpr (UseTileReady)
                {
                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
                    {
                        if (tid == 0)
                        {
                            ag_wait_signal_ge(rank_tile_ready_flags,
                                              tile_m,
                                              signal_value);
                        }
                    }
                    else if (tid == 0)
                    {
                        for (int src_rank = segment_start;
                             src_rank <= segment_end;
                             ++src_rank)
                        {
                            ag_wait_signal_ge(rank_signals,
                                              src_rank,
                                              signal_value);
                        }
                        if (rank_tile_ready_flags != nullptr)
                        {
                            __threadfence_system();
                            atomicExch(rank_tile_ready_flags + tile_m,
                                       signal_value);
                        }
                    }
                }
                else if (tid == 0)
                {
                    for (int src_rank = segment_start; src_rank <= segment_end;
                         ++src_rank)
                    {
                        ag_wait_signal_ge(rank_signals,
                                          src_rank,
                                          signal_value);
                    }
                }
                __syncthreads();
            }
        }
    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc0[N_MMACS_PER_WAVE];
    ag_float4_t acc1[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc0[nn]);
        zero_vec4(acc1[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        for (int load = tid; load < A_LOADS; load += blockDim.x)
        {
            const int m = load / (BK / A_LOAD_VEC);
            const int kk = (load - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                const int k = kk + i;
                const int row_group = m >> 4;
                const int row16 = m & 15;
                const int slot =
                    row_group * FRAGMENT_ELEMS +
                    (UseDsReadSwizzle
                         ? ag_dsread_m32x16_b16_swizzled_physical_slot(
                               ag_dsread_m32x16_b16_fragment_slot(row16, k))
                         : ag_dsread_m32x16_b16_fragment_slot(row16, k));
                sh_a[slot] = value[i];
            }
        }

        for (int load = tid; load < B_LOADS_VEC; load += blockDim.x)
        {
            const int kk = load / (BN / B_LOAD_VEC);
            const int n_vec = load - kk * (BN / B_LOAD_VEC);
            const int n_off = n_vec * B_LOAD_VEC;
            const int n = n_base + n_off;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (k0 + kk < k_dim && n < n_dim)
            {
                const int64_t base =
                    (static_cast<int64_t>(expert) * k_dim + k0 + kk) *
                        n_dim +
                    n;
                if (n + B_LOAD_VEC <= n_dim)
                {
                    value =
                        *reinterpret_cast<const ag_bf16x4_t *>(weight + base);
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < B_LOAD_VEC; ++i)
                    {
                        value[i] = (n + i < n_dim) ? weight[base + i] : 0;
                    }
                }
            }
            const int n_tile = n_off >> 4;
            const int n16 = n_off & 15;
            if constexpr (UseDsReadSwizzle)
            {
#pragma unroll
                for (int i = 0; i < B_LOAD_VEC; ++i)
                {
                    const int slot =
                        n_tile * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_swizzled_physical_slot(
                            ag_dsread_m32x16_b16_fragment_slot(n16 + i, kk));
                    sh_b[slot] = value[i];
                }
            }
            else
            {
                const int slot =
                    n_tile * FRAGMENT_ELEMS +
                    ag_dsread_m32x16_b16_fragment_slot(n16, kk);
                *reinterpret_cast<ag_bf16x4_t *>(sh_b + slot) = value;
            }
        }

        __syncthreads();

        const int lane_matrix_offset =
            ag_dsread_m32x16_lane_offset_elems(lane);
        const int ds_lane_offset =
            UseDsReadSwizzle
                ? ag_dsread_m32x16_b16_swizzled_lane_offset_elems(lane)
                : lane_matrix_offset;
        const int row_group0 = wave_m * 2;
        const uint32_t a0_lds_offset =
            static_cast<uint32_t>((row_group0 * FRAGMENT_ELEMS +
                                   ds_lane_offset) *
                                  sizeof(uint16_t));
        const uint32_t a1_lds_offset =
            static_cast<uint32_t>(((row_group0 + 1) * FRAGMENT_ELEMS +
                                   ds_lane_offset) *
                                  sizeof(uint16_t));
        ag_uint4_t ra0;
        ag_uint4_t ra1;
        ag_ds_read_m32x16_b16(ra0, a0_lds_offset);
        ag_ds_read_m32x16_b16(ra1, a1_lds_offset);

#pragma unroll
        for (int nn0 = 0; nn0 < N_MMACS_PER_WAVE; nn0 += 2)
        {
            const int n_tile0 = wave_n_chunk * N_MMACS_PER_WAVE + nn0;
            const int n_tile1 = n_tile0 + 1;
            const uint32_t b0_lds_offset =
                static_cast<uint32_t>((LDS_A_ELEMS +
                                       n_tile0 * FRAGMENT_ELEMS +
                                       ds_lane_offset) *
                                      sizeof(uint16_t));
            const uint32_t b1_lds_offset =
                static_cast<uint32_t>((LDS_A_ELEMS +
                                       n_tile1 * FRAGMENT_ELEMS +
                                       ds_lane_offset) *
                                      sizeof(uint16_t));
            ag_uint4_t rb0;
            ag_uint4_t rb1;
            ag_ds_read_m32x16_b16(rb0, b0_lds_offset);
            ag_ds_read_m32x16_b16(rb1, b1_lds_offset);
            ag_wait_lgkmcnt0();

            const ag_bf16x4_t a00 = ag_bf16x4_from_u32x2(ra0, 0);
            const ag_bf16x4_t a01 = ag_bf16x4_from_u32x2(ra0, 1);
            const ag_bf16x4_t a10 = ag_bf16x4_from_u32x2(ra1, 0);
            const ag_bf16x4_t a11 = ag_bf16x4_from_u32x2(ra1, 1);
            const ag_bf16x4_t b00 = ag_bf16x4_from_u32x2(rb0, 0);
            const ag_bf16x4_t b01 = ag_bf16x4_from_u32x2(rb0, 1);
            const ag_bf16x4_t b10 = ag_bf16x4_from_u32x2(rb1, 0);
            const ag_bf16x4_t b11 = ag_bf16x4_from_u32x2(rb1, 1);

            acc0[nn0 + 0] = ag_mmac_bf16(a00, b00, acc0[nn0 + 0]);
            acc0[nn0 + 0] = ag_mmac_bf16(a01, b01, acc0[nn0 + 0]);
            acc0[nn0 + 1] = ag_mmac_bf16(a00, b10, acc0[nn0 + 1]);
            acc0[nn0 + 1] = ag_mmac_bf16(a01, b11, acc0[nn0 + 1]);
            acc1[nn0 + 0] = ag_mmac_bf16(a10, b00, acc1[nn0 + 0]);
            acc1[nn0 + 0] = ag_mmac_bf16(a11, b01, acc1[nn0 + 0]);
            acc1[nn0 + 1] = ag_mmac_bf16(a10, b10, acc1[nn0 + 1]);
            acc1[nn0 + 1] = ag_mmac_bf16(a11, b11, acc1[nn0 + 1]);
        }
        __syncthreads();
    }

    const int row0 = row_base + lane_row16;
    const int row1 = row0 + 16;
    const int flat0 =
        (row0 < tile_rows) ? sorted_token_ids[sorted_start + row0] : -1;
    const int flat1 =
        (row1 < tile_rows) ? sorted_token_ids[sorted_start + row1] : -1;
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            const int n = col_base + nn * 16 + lane_col4 + i * 4;
            if (flat0 >= 0 && flat0 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat0) * n_dim + n] =
                    ag_float_to_bf16_bits(acc0[nn][i]);
            }
            if (flat1 >= 0 && flat1 < full_m * topk && n < n_dim)
            {
                output[static_cast<int64_t>(flat1) * n_dim + n] =
                    ag_float_to_bf16_bits(acc1[nn][i]);
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

template <bool UseTileReady,
          bool UseDsReadFragmentLayout,
          bool UseDsReadSwizzle = false>
__launch_bounds__(1024, 1) __global__
    void ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel(
        const uint16_t *full_input,
        const uint16_t *weight,
        const int *sorted_token_ids,
        const int *sorted_gather_index,
        const int *rank_tile_info,
        const int *rank_tile_problem_info,
        const int *ntiles,
        const int *rank_signals,
        int *rank_tile_ready_flags,
        int *n_chunk_done_counts,
        int *n_chunk_done_flags,
        uint16_t *output,
        int full_m,
        int n_dim,
        int k_dim,
        int topk,
        int world_size,
        int signal_value,
        int signal_chunk_rows,
        int chunks_per_rank)
{
    constexpr int BM = 128;
    constexpr int BN = 256;
    constexpr int BK = 32;
    constexpr int MMAC_K = 16;
    constexpr int WARP_SIZE = 64;
    constexpr int WAVES_PER_BLOCK = 16;
    constexpr int N_CHUNKS = 2;
    constexpr int WAVE_M = 16;
    constexpr int N_PER_WAVE = BN / N_CHUNKS;
    constexpr int N_MMACS_PER_WAVE = N_PER_WAVE / 16;
    constexpr int A_LOAD_VEC = 4;
    constexpr int B_LOAD_VEC = 4;
    constexpr int A_LOADS = BM * (BK / A_LOAD_VEC);
    constexpr int B_LOADS_VEC = BK * (BN / B_LOAD_VEC);
    constexpr int FRAGMENT_ELEMS = 16 * BK;
    constexpr int LDS_A_ELEMS = BK * BM;
    constexpr int LDS_B_ELEMS = BK * BN;

    const int tile_m = blockIdx.x;
    const int n_base = blockIdx.y * BN;
    if (tile_m >= *ntiles)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int wave_idx = tid / WARP_SIZE;
    const int lane = tid - wave_idx * WARP_SIZE;
    const int wave_m = wave_idx >> 1;
    const int wave_n_chunk = wave_idx & 1;
    const int row_base = wave_m * WAVE_M;
    const int col_base = n_base + wave_n_chunk * N_PER_WAVE;
    const int lane_row16 = lane & 15;
    const int lane_col4 = lane >> 4;

    const int *tile_info =
        rank_tile_problem_info != nullptr
            ? rank_tile_problem_info +
                  tile_m * AG_RANK_TILE_PROBLEM_INFO_STRIDE
            : rank_tile_info + tile_m * AG_RANK_TILE_INFO_STRIDE;
    const int expert = tile_info[AG_RANK_TILE_EXPERT];
    const int sorted_start = tile_info[AG_RANK_TILE_SORTED_START];
    const int tile_rows = tile_info[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const bool use_chunk_signal =
            signal_chunk_rows > 0 && chunks_per_rank > 0;
        if (use_chunk_signal)
        {
            if (rank_tile_problem_info != nullptr)
            {
                const int chunk_start =
                    tile_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
                const int chunk_end = tile_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
                if (chunk_start >= 0 && chunk_end >= chunk_start)
                {
                    for (int chunk = chunk_start + tid; chunk <= chunk_end;
                         chunk += blockDim.x)
                    {
                        ag_wait_signal_ge(rank_signals, chunk, signal_value);
                    }
                }
            }
            else
            {
                for (int row = tid; row < tile_rows; row += blockDim.x)
                {
                    const int token = sorted_gather_index[sorted_start + row];
                    if (token >= 0 && token < full_m)
                    {
                        const int chunk_id =
                            ag_token_chunk_id(token,
                                              full_m,
                                              world_size,
                                              signal_chunk_rows,
                                              chunks_per_rank);
                        ag_wait_signal_ge(rank_signals, chunk_id, signal_value);
                    }
                }
            }
            __syncthreads();
        }
        else
        {
            const int segment_start = tile_info[AG_RANK_TILE_RANK_START];
            const int segment_end = tile_info[AG_RANK_TILE_RANK_END];
            if (segment_start >= 0 && segment_end < world_size)
            {
                if constexpr (UseTileReady)
                {
                    if (rank_tile_ready_flags != nullptr && blockIdx.y != 0)
                    {
                        if (tid == 0)
                        {
                            ag_wait_signal_ge(rank_tile_ready_flags,
                                              tile_m,
                                              signal_value);
                        }
                    }
                    else if (tid == 0)
                    {
                        for (int src_rank = segment_start;
                             src_rank <= segment_end;
                             ++src_rank)
                        {
                            ag_wait_signal_ge(rank_signals,
                                              src_rank,
                                              signal_value);
                        }
                        if (rank_tile_ready_flags != nullptr)
                        {
                            __threadfence_system();
                            atomicExch(rank_tile_ready_flags + tile_m,
                                       signal_value);
                        }
                    }
                }
                else if (tid == 0)
                {
                    for (int src_rank = segment_start; src_rank <= segment_end;
                         ++src_rank)
                    {
                        ag_wait_signal_ge(rank_signals,
                                          src_rank,
                                          signal_value);
                    }
                }
                __syncthreads();
            }
        }
    }

    __shared__ uint16_t shmem[LDS_A_ELEMS + LDS_B_ELEMS];
    uint16_t *sh_a = shmem;
    uint16_t *sh_b = shmem + LDS_A_ELEMS;

    ag_float4_t acc[N_MMACS_PER_WAVE];
#pragma unroll
    for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
    {
        zero_vec4(acc[nn]);
    }

    for (int k0 = 0; k0 < k_dim; k0 += BK)
    {
        if (tid < A_LOADS)
        {
            const int m = tid / (BK / A_LOAD_VEC);
            const int kk = (tid - m * (BK / A_LOAD_VEC)) * A_LOAD_VEC;
            const int token_a =
                (m < tile_rows) ? sorted_gather_index[sorted_start + m] : -1;
            const bool valid_a = token_a >= 0 && token_a < full_m;
            const int gk = k0 + kk;
            ag_bf16x4_t value;
            zero_vec4(value);
            if (valid_a && gk < k_dim)
            {
                value = load_bf16x4_k_contiguous(
                    full_input,
                    static_cast<int64_t>(token_a) * k_dim + gk,
                    gk,
                    k_dim);
            }
#pragma unroll
            for (int i = 0; i < A_LOAD_VEC; ++i)
            {
                if constexpr (UseDsReadSwizzle)
                {
                    const int k = kk + i;
                    const int row_group = m >> 4;
                    const int row16 = m & 15;
                    const int slot =
                        row_group * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_swizzled_physical_slot(
                            ag_dsread_m32x16_b16_fragment_slot(row16, k));
                    sh_a[slot] = value[i];
                }
                else if constexpr (UseDsReadFragmentLayout)
                {
                    const int k = kk + i;
                    const int row_group = m >> 4;
                    const int row16 = m & 15;
                    const int slot =
                        row_group * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_fragment_slot(row16, k);
                    sh_a[slot] = value[i];
                }
                else
                {
                    sh_a[(kk + i) * BM + m] = value[i];
                }
            }
        }

        if constexpr (UseDsReadFragmentLayout)
        {
            for (int load = tid; load < B_LOADS_VEC; load += blockDim.x)
            {
                const int kk = load / (BN / B_LOAD_VEC);
                const int n_vec = load - kk * (BN / B_LOAD_VEC);
                const int n_off = n_vec * B_LOAD_VEC;
                const int n = n_base + n_off;
                ag_bf16x4_t value;
                zero_vec4(value);
                if (k0 + kk < k_dim && n < n_dim)
                {
                    const int64_t base =
                        (static_cast<int64_t>(expert) * k_dim + k0 + kk) *
                            n_dim +
                        n;
                    if (n + B_LOAD_VEC <= n_dim)
                    {
                        value = *reinterpret_cast<const ag_bf16x4_t *>(
                            weight + base);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < B_LOAD_VEC; ++i)
                        {
                            value[i] = (n + i < n_dim) ? weight[base + i] : 0;
                        }
                    }
                }
                const int n_tile = n_off >> 4;
                const int n16 = n_off & 15;
                if constexpr (UseDsReadSwizzle)
                {
#pragma unroll
                    for (int i = 0; i < B_LOAD_VEC; ++i)
                    {
                        const int slot =
                            n_tile * FRAGMENT_ELEMS +
                            ag_dsread_m32x16_b16_swizzled_physical_slot(
                                ag_dsread_m32x16_b16_fragment_slot(n16 + i,
                                                                   kk));
                        sh_b[slot] = value[i];
                    }
                }
                else
                {
                    const int slot =
                        n_tile * FRAGMENT_ELEMS +
                        ag_dsread_m32x16_b16_fragment_slot(n16, kk);
                    *reinterpret_cast<ag_bf16x4_t *>(sh_b + slot) = value;
                }
            }
        }
        else
        {
            for (int idx = tid; idx < LDS_B_ELEMS; idx += blockDim.x)
            {
                const int kk = idx / BN;
                const int n_off = idx - kk * BN;
                const int n = n_base + n_off;
                uint16_t value = 0;
                if (k0 + kk < k_dim && n < n_dim)
                {
                    value =
                        weight[(static_cast<int64_t>(expert) * k_dim + k0 +
                                kk) *
                                   n_dim +
                               n];
                }
                sh_b[idx] = value;
            }
        }

        __syncthreads();

        if constexpr (UseDsReadFragmentLayout)
        {
            const int lane_matrix_offset =
                ag_dsread_m32x16_lane_offset_elems(lane);
            const int a_lane_matrix_offset =
                UseDsReadSwizzle
                    ? ag_dsread_m32x16_b16_swizzled_lane_offset_elems(lane)
                    : lane_matrix_offset;
            const uint32_t a_lds_offset =
                static_cast<uint32_t>((wave_m * FRAGMENT_ELEMS +
                                       a_lane_matrix_offset) *
                                      sizeof(uint16_t));
            ag_uint4_t ra;
            ag_ds_read_m32x16_b16(ra, a_lds_offset);

#pragma unroll
            for (int nn0 = 0; nn0 < N_MMACS_PER_WAVE; nn0 += 2)
            {
                const int n_tile0 = wave_n_chunk * N_MMACS_PER_WAVE + nn0;
                const int b_lane_matrix_offset =
                    UseDsReadSwizzle
                        ? ag_dsread_m32x16_b16_swizzled_lane_offset_elems(lane)
                        : lane_matrix_offset;
                const uint32_t b0_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           n_tile0 * FRAGMENT_ELEMS +
                                           b_lane_matrix_offset) *
                                          sizeof(uint16_t));
                const int n_tile1 = n_tile0 + 1;
                const uint32_t b1_lds_offset =
                    static_cast<uint32_t>((LDS_A_ELEMS +
                                           n_tile1 * FRAGMENT_ELEMS +
                                           b_lane_matrix_offset) *
                                          sizeof(uint16_t));
                ag_uint4_t rb0;
                ag_uint4_t rb1;
                ag_ds_read_m32x16_b16(rb0, b0_lds_offset);
                ag_ds_read_m32x16_b16(rb1, b1_lds_offset);
                ag_wait_lgkmcnt0();

                const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                const ag_bf16x4_t a1 = ag_bf16x4_from_u32x2(ra, 1);

                const ag_bf16x4_t b00 = ag_bf16x4_from_u32x2(rb0, 0);
                const ag_bf16x4_t b01 = ag_bf16x4_from_u32x2(rb0, 1);
                const ag_bf16x4_t b10 = ag_bf16x4_from_u32x2(rb1, 0);
                const ag_bf16x4_t b11 = ag_bf16x4_from_u32x2(rb1, 1);
                acc[nn0 + 0] = ag_mmac_bf16(a0, b00, acc[nn0 + 0]);
                acc[nn0 + 0] = ag_mmac_bf16(a1, b01, acc[nn0 + 0]);
                acc[nn0 + 1] = ag_mmac_bf16(a0, b10, acc[nn0 + 1]);
                acc[nn0 + 1] = ag_mmac_bf16(a1, b11, acc[nn0 + 1]);
            }
        }
        else
        {
            const int lane_k = lane / 4;
            const int lane_group = (lane & 3) * 8;
#pragma unroll
            for (int kk0 = 0; kk0 < BK; kk0 += MMAC_K)
            {
                const uint32_t a_lds_offset =
                    static_cast<uint32_t>(((kk0 + lane_k) * BM + row_base +
                                           lane_group) *
                                          sizeof(uint16_t));
                ag_uint4_t ra;
                ag_ds_read_m32x16_b16(ra, a_lds_offset);

#pragma unroll
                for (int nn32 = 0; nn32 < N_MMACS_PER_WAVE / 2; ++nn32)
                {
                    const uint32_t b_lds_offset =
                        static_cast<uint32_t>((LDS_A_ELEMS +
                                               (kk0 + lane_k) * BN +
                                               wave_n_chunk * N_PER_WAVE +
                                               nn32 * 32 + lane_group) *
                                              sizeof(uint16_t));
                    ag_uint4_t rb;
                    ag_ds_read_m32x16_b16(rb, b_lds_offset);
                    ag_wait_lgkmcnt0();

                    const ag_bf16x4_t a0 = ag_bf16x4_from_u32x2(ra, 0);
                    const ag_bf16x4_t b0 = ag_bf16x4_from_u32x2(rb, 0);
                    const ag_bf16x4_t b1 = ag_bf16x4_from_u32x2(rb, 1);
                    acc[nn32 * 2 + 0] =
                        ag_mmac_bf16(a0, b0, acc[nn32 * 2 + 0]);
                    acc[nn32 * 2 + 1] =
                        ag_mmac_bf16(a0, b1, acc[nn32 * 2 + 1]);
                }
            }
        }
        __syncthreads();
    }

    const int row = row_base + lane_row16;
    const int flat_row =
        (row < tile_rows) ? sorted_token_ids[sorted_start + row] : -1;
    if (flat_row >= 0 && flat_row < full_m * topk)
    {
#pragma unroll
        for (int nn = 0; nn < N_MMACS_PER_WAVE; ++nn)
        {
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int n = col_base + nn * 16 + lane_col4 + i * 4;
                if (n < n_dim)
                {
                    output[static_cast<int64_t>(flat_row) * n_dim + n] =
                        ag_float_to_bf16_bits(acc[nn][i]);
                }
            }
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && n_chunk_done_counts != nullptr &&
        n_chunk_done_flags != nullptr)
    {
        __threadfence_system();
        const int done = atomicAdd(n_chunk_done_counts + blockIdx.y, 1) + 1;
        if (done == *ntiles)
        {
            n_chunk_done_flags[blockIdx.y] = signal_value;
        }
    }
}

template <typename T>
__global__ void pack_weight_kn_to_nk_kernel(const T *weight,
                                            T *packed_weight,
                                            int k_dim,
                                            int n_dim)
{
    constexpr int tile_dim = 32;
    constexpr int block_rows = 8;
    __shared__ T tile[tile_dim][tile_dim + 1];

    const int expert = blockIdx.z;
    const int src_n = blockIdx.x * tile_dim + threadIdx.x;
    const int src_k = blockIdx.y * tile_dim + threadIdx.y;
    for (int row = 0; row < tile_dim; row += block_rows)
    {
        const int k = src_k + row;
        if (src_n < n_dim && k < k_dim)
        {
            tile[threadIdx.y + row][threadIdx.x] =
                weight[(static_cast<int64_t>(expert) * k_dim + k) * n_dim +
                       src_n];
        }
    }
    __syncthreads();

    const int dst_k = blockIdx.y * tile_dim + threadIdx.x;
    const int dst_n = blockIdx.x * tile_dim + threadIdx.y;
    for (int row = 0; row < tile_dim; row += block_rows)
    {
        const int n = dst_n + row;
        if (dst_k < k_dim && n < n_dim)
        {
            packed_weight[(static_cast<int64_t>(expert) * n_dim + n) * k_dim +
                          dst_k] = tile[threadIdx.x][threadIdx.y + row];
        }
    }
}

__global__ void wait_rank_signals_kernel(const int *rank_signals,
                                         int world_size,
                                         int signal_value)
{
    if (threadIdx.x != 0 || blockIdx.x != 0 || rank_signals == nullptr)
    {
        return;
    }
    for (int src_rank = 0; src_rank < world_size; ++src_rank)
    {
        const volatile int *signal =
            reinterpret_cast<const volatile int *>(rank_signals + src_rank);
        while (*signal < signal_value)
        {
            __builtin_amdgcn_s_sleep(1);
        }
    }
}

__global__ void wait_rank_signals_system_kernel(const int *rank_signals,
                                                int world_size,
                                                int signal_value)
{
    if (threadIdx.x != 0 || blockIdx.x != 0 || rank_signals == nullptr)
    {
        return;
    }
    for (int src_rank = 0; src_rank < world_size; ++src_rank)
    {
        while (__hip_atomic_load(rank_signals + src_rank,
                                 __ATOMIC_ACQUIRE,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < signal_value)
        {
            __builtin_amdgcn_s_sleep(1);
        }
    }
}

template <typename T>
void launch_reference_forward(const void *full_input,
                              const void *weight,
                              const void *topk_ids,
                              void *output,
                              int full_m,
                              int n_dim,
                              int k_dim,
                              int num_experts,
                              int topk,
                              c10::ScalarType index_dtype,
                              hipStream_t stream)
{
    const int threads = 256;
    const int64_t total = static_cast<int64_t>(full_m) * topk * n_dim;
    const dim3 grid((total + threads - 1) / threads);
    if (index_dtype == c10::ScalarType::Long)
    {
        hipLaunchKernelGGL((ag_group_gemm_reference_kernel<T, int64_t>),
                           grid,
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const T *>(full_input),
                           static_cast<const T *>(weight),
                           topk_ids,
                           static_cast<T *>(output),
                           full_m,
                           n_dim,
                           k_dim,
                           num_experts,
                           topk);
    }
    else
    {
        hipLaunchKernelGGL((ag_group_gemm_reference_kernel<T, int32_t>),
                           grid,
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const T *>(full_input),
                           static_cast<const T *>(weight),
                           topk_ids,
                           static_cast<T *>(output),
                           full_m,
                           n_dim,
                           k_dim,
                           num_experts,
                           topk);
    }
}

} // namespace

void ag_group_gemm_update_rank_m_offsets_impl(int *device_offsets,
                                              const int *host_offsets,
                                              int count,
                                              hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(device_offsets != nullptr);
    FTRAIN_HYGON_CHECK(host_offsets != nullptr);
    FTRAIN_HYGON_CHECK(count > 0 && count <= MAX_LOCAL_WORLD_SIZE + 1);
    AgRankMOffsets offsets{};
    std::copy(host_offsets, host_offsets + count, offsets.values);
    hipLaunchKernelGGL(update_rank_m_offsets_kernel,
                       dim3(1),
                       dim3(MAX_LOCAL_WORLD_SIZE + 1),
                       0,
                       stream,
                       device_offsets,
                       offsets,
                       count);
    CUDA_CHECK(hipGetLastError());
}

void ag_group_gemm_prepare_metadata_impl(const void *topk_ids,
                                         int *sorted_token_ids,
                                         int *sorted_gather_index,
                                         int *expert_rank_counts,
                                         int *expert_rank_counters,
                                         int *expert_rank_offsets,
                                         int *global_expert_offsets,
                                         int *rank_cursors,
                                         int *rank_tile_info,
                                         int *tile_experts,
	                                         int *tile_starts,
	                                         int *tile_rank_starts,
	                                         int *tile_rank_ends,
	                                         int *rank_tile_problem_info,
	                                         int *ntiles,
	                                         const int *rank_m_offsets,
	                                         int full_m,
	                                         int topk,
	                                         int num_experts,
	                                         int world_size,
	                                         int rank,
	                                         int sorted_capacity,
	                                         int rank_tile_capacity,
	                                         int signal_chunk_rows,
	                                         int chunks_per_rank,
	                                         bool strict_rank_tile,
                                         bool split_local_rank,
                                         bool build_problem_info,
	                                         bool stage_signal_layout,
	                                         bool stage_signal_zigzag,
	                                         c10::ScalarType index_dtype,
	                                         hipStream_t stream)
{
    const int BM =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT",
                           false)
            ? 256
            : 128;
    const int threads = 256;
    const int total = full_m * topk;
    const dim3 flat_grid((total + threads - 1) / threads);
    const dim3 sorted_grid((sorted_capacity + threads - 1) / threads);
    const int num_buckets = num_experts * world_size;
    const bool stable_expert_rank_sort = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_STABLE_EXPERT_RANK_SORT", false);
    const bool nn_direct_chunk_signal = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_V36_NN_CHUNK_SIGNAL", false);

    CUDA_CHECK(hipMemsetAsync(
        expert_rank_counts, 0, sizeof(int) * num_buckets, stream));
    CUDA_CHECK(hipMemsetAsync(
        expert_rank_counters, 0, sizeof(int) * num_buckets, stream));
    CUDA_CHECK(hipMemsetAsync(ntiles, 0, sizeof(int), stream));

    const bool ragged_m = rank_m_offsets != nullptr;
    if (index_dtype == c10::ScalarType::Long && ragged_m)
    {
        hipLaunchKernelGGL((count_topk_by_expert_rank_kernel<int64_t, true>),
                           flat_grid,
                           dim3(threads),
                           0,
                           stream,
                           topk_ids,
                           expert_rank_counts,
                           full_m,
                           topk,
                           num_experts,
                           world_size,
                           rank,
                           stage_signal_layout,
                           stage_signal_zigzag,
                           rank_m_offsets);
    }
    else if (index_dtype == c10::ScalarType::Long)
    {
        hipLaunchKernelGGL((count_topk_by_expert_rank_kernel<int64_t, false>),
                           flat_grid,
                           dim3(threads),
                           0,
                           stream,
                           topk_ids,
                           expert_rank_counts,
                           full_m,
                           topk,
                           num_experts,
                           world_size,
                           rank,
                           stage_signal_layout,
                           stage_signal_zigzag,
                           nullptr);
    }
    else if (ragged_m)
    {
        hipLaunchKernelGGL((count_topk_by_expert_rank_kernel<int32_t, true>),
                           flat_grid,
                           dim3(threads),
                           0,
                           stream,
                           topk_ids,
                           expert_rank_counts,
                           full_m,
                           topk,
                           num_experts,
                           world_size,
                           rank,
                           stage_signal_layout,
                           stage_signal_zigzag,
                           rank_m_offsets);
    }
    else
    {
        hipLaunchKernelGGL((count_topk_by_expert_rank_kernel<int32_t, false>),
                           flat_grid,
                           dim3(threads),
                           0,
                           stream,
                           topk_ids,
                           expert_rank_counts,
                           full_m,
                           topk,
                           num_experts,
                           world_size,
                           rank,
                           stage_signal_layout,
                           stage_signal_zigzag,
                           nullptr);
    }

    hipLaunchKernelGGL(build_expert_rank_offsets_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       expert_rank_counts,
                       expert_rank_offsets,
                       global_expert_offsets,
                       rank_cursors,
                       ntiles,
                       num_experts,
                       world_size,
                       BM);

    hipLaunchKernelGGL(init_sorted_token_ids_kernel,
                       sorted_grid,
                       dim3(threads),
                       0,
                           stream,
                           sorted_token_ids,
                           sorted_gather_index,
                           sorted_capacity);

    if (stable_expert_rank_sort)
    {
        const dim3 stable_grid(num_experts);
        const dim3 stable_block(64);
        if (index_dtype == c10::ScalarType::Long && ragged_m)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_stable_kernel<
                    int64_t, true>),
                stable_grid,
                stable_block,
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                rank_m_offsets);
        }
        else if (index_dtype == c10::ScalarType::Long)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_stable_kernel<
                    int64_t, false>),
                stable_grid,
                stable_block,
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                nullptr);
        }
        else if (ragged_m)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_stable_kernel<
                    int32_t, true>),
                stable_grid,
                stable_block,
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                rank_m_offsets);
        }
        else
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_stable_kernel<
                    int32_t, false>),
                stable_grid,
                stable_block,
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                nullptr);
        }
    }
    else
    {
        CUDA_CHECK(hipMemsetAsync(
            expert_rank_counters, 0, sizeof(int) * num_buckets, stream));
        if (index_dtype == c10::ScalarType::Long && ragged_m)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_kernel<int64_t, true>),
                flat_grid,
                dim3(threads),
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_counters,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                rank_m_offsets);
        }
        else if (index_dtype == c10::ScalarType::Long)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_kernel<int64_t, false>),
                flat_grid,
                dim3(threads),
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_counters,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                nullptr);
        }
        else if (ragged_m)
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_kernel<int32_t, true>),
                flat_grid,
                dim3(threads),
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_counters,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                rank_m_offsets);
        }
        else
        {
            hipLaunchKernelGGL(
                (scatter_sorted_token_ids_by_expert_rank_kernel<int32_t, false>),
                flat_grid,
                dim3(threads),
                0,
                stream,
                topk_ids,
                sorted_token_ids,
                sorted_gather_index,
                expert_rank_counters,
                expert_rank_offsets,
                full_m,
                topk,
                num_experts,
                world_size,
                rank,
                stage_signal_layout,
                stage_signal_zigzag,
                nullptr);
        }
    }

    hipLaunchKernelGGL(build_expert_rank_tile_descriptors_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       rank_cursors,
                       global_expert_offsets,
                       rank_tile_info,
                       tile_experts,
                       tile_starts,
                       tile_rank_starts,
                       tile_rank_ends,
                       ntiles,
                       num_experts,
                       world_size,
                       rank,
                       strict_rank_tile,
                       split_local_rank,
                       stage_signal_layout,
                       BM);
	    if (build_problem_info && rank_tile_problem_info != nullptr)
	    {
	        hipLaunchKernelGGL(build_rank_tile_problem_info_kernel,
	                           dim3(rank_tile_capacity),
	                           dim3(128),
	                           0,
	                           stream,
	                           rank_tile_info,
	                           sorted_gather_index,
	                           ntiles,
	                           rank_tile_problem_info,
	                           full_m,
	                           world_size,
                           signal_chunk_rows,
                           chunks_per_rank,
                           rank_tile_capacity,
                           nn_direct_chunk_signal,
                           rank_m_offsets,
                            ragged_m,
                            rank,
                            stage_signal_layout);
	    }
}

struct __attribute__((packed)) AgMt256ContractRankTileArgs
{
    const uint16_t *input_buffer;
    const uint16_t *weight;
    const int32_t *sorted_expert_offsets;
    uint16_t *output;
    const int32_t *barrier_ptr;
    const int32_t *tile_info;
    uint32_t problem_count;
    uint32_t n_dim;
    uint32_t k_dim;
    uint32_t topk;
    uint32_t rank;
    uint32_t wait_signal_value;
    uint32_t epilogue_mode;
    uint32_t chunk_size_n;
    uint32_t num_done_groups;
    uint32_t reserved;
    void *topk_stage;
    int32_t *chunk_done_counters;
    int32_t *chunk_done_flags;
};

static_assert(sizeof(AgMt256ContractRankTileArgs) == 112,
              "MT256 native contract args must stay 112 bytes");

struct AgMt256Scratch
{
    void *ptr = nullptr;
    size_t bytes = 0;
};

thread_local AgMt256Scratch g_mt256_native_packed_input;
thread_local AgMt256Scratch g_mt256_native_packed_output;
thread_local AgMt256Scratch g_mt256_native_tile_info;
thread_local AgMt256Scratch g_mt256_native_expert_offsets;
thread_local AgMt256Scratch g_mt256_v6_main_tile_info;
thread_local AgMt256Scratch g_mt256_v6_tail_tile_info;
thread_local AgMt256Scratch g_mt256_v6_split_counts;
thread_local AgMt256Scratch g_mt256_v6_stage_cursors;
thread_local AgMt256Scratch g_mt256_v11_stage_tile_info;
thread_local AgMt256Scratch g_mt256_v11_split_counts;
thread_local AgMt256Scratch g_mt256_v11_stage_cursors;
thread_local AgMt256Scratch g_mt256_v11_window_offsets;
thread_local AgMt256Scratch g_mt256_v11_chunk_tile_info;
thread_local AgMt256Scratch g_mt256_v11_chunk_problem_info;
thread_local AgMt256Scratch g_mt256_v11_chunk_ready_offsets;
thread_local AgMt256Scratch g_mt256_v12_task_claims;
thread_local AgMt256Scratch g_mt256_v36_direct_control;

struct AgMt256V6ProblemCountCache
{
    const int *rank_tile_info = nullptr;
    const int *ntiles = nullptr;
    int max_tiles = 0;
    int problem_count = 0;
    bool valid = false;
};

thread_local AgMt256V6ProblemCountCache g_mt256_v6_problem_count_cache;

struct AgMt256V6HybridCache
{
    const int *rank_tile_info = nullptr;
    const int *ntiles = nullptr;
    const void *main_tile_info = nullptr;
    const void *tail_tile_info = nullptr;
    int max_tail_rows = 0;
    int max_tiles = 0;
    int world_size = 0;
    int rank = 0;
    bool stage_major = false;
    int original_problem_count = 0;
    int main_problem_count = 0;
    int tail_problem_count = 0;
    bool valid = false;
};

thread_local AgMt256V6HybridCache g_mt256_v6_hybrid_cache;

struct AgMt256V11StageCache
{
    const int *rank_tile_info = nullptr;
    const int *ntiles = nullptr;
    const void *stage_tile_info = nullptr;
    int max_tiles = 0;
    int world_size = 0;
    int rank = 0;
    int early_stages = 0;
    int prefix_experts = 0;
    int window_experts = 0;
    bool stage_major = false;
    bool window_stage_major = false;
    bool split_launch = false;
    int problem_count = 0;
    int first_problem_count = 0;
    bool valid = false;
};

thread_local AgMt256V11StageCache g_mt256_v11_stage_cache;

struct AgMt256V11ChunkCache
{
    const int *rank_tile_problem_info = nullptr;
    const int *ntiles = nullptr;
    const void *chunk_tile_info = nullptr;
    const void *chunk_problem_info = nullptr;
    int max_tiles = 0;
    int world_size = 0;
    int rank = 0;
    int chunks_per_rank = 0;
    int prefix_experts = 0;
    bool stage_major = false;
    int problem_count = 0;
    bool valid = false;
};

thread_local AgMt256V11ChunkCache g_mt256_v11_chunk_cache;

bool ag_mt256_ensure_scratch(AgMt256Scratch *scratch, size_t bytes, void **ptr)
{
    *ptr = nullptr;
    if (scratch == nullptr || bytes == 0)
    {
        return false;
    }
    if (scratch->ptr != nullptr && scratch->bytes >= bytes)
    {
        *ptr = scratch->ptr;
        return true;
    }
    if (scratch->ptr != nullptr)
    {
        hipFree(scratch->ptr);
        scratch->ptr = nullptr;
        scratch->bytes = 0;
    }
    hipError_t err = hipMalloc(&scratch->ptr, bytes);
    if (err != hipSuccess)
    {
        scratch->ptr = nullptr;
        scratch->bytes = 0;
        return false;
    }
    scratch->bytes = bytes;
    *ptr = scratch->ptr;
    return true;
}

__global__ void ag_mt256_count_v6_hybrid_tile_lists_kernel(
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ ntiles,
    int *__restrict__ stage_counts,
    int max_tail_rows,
    int world_size,
    int rank,
    bool stage_major,
    int max_tiles)
{
    const int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem >= max_tiles || problem >= *ntiles)
    {
        return;
    }

    constexpr int kTailM = 64;
    const int *src = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int rows = src[AG_RANK_TILE_ROWS];
    if (rows <= 0)
    {
        return;
    }
    const int stage =
        stage_major
            ? ag_moe_tile_stage(
                  src[AG_RANK_TILE_RANK_END], rank, world_size)
            : 0;
    const bool use_tail = rows < 256 && rows <= max_tail_rows;
    const int type_offset = use_tail ? MAX_LOCAL_WORLD_SIZE : 0;
    const int descriptors = use_tail ? (rows + kTailM - 1) / kTailM : 1;
    atomicAdd(stage_counts + type_offset + stage, descriptors);
}

__global__ void ag_mt256_prefix_v6_hybrid_tile_lists_kernel(
    int *__restrict__ stage_cursors,
    int *__restrict__ split_counts,
    int world_size)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    for (int type = 0; type < 2; ++type)
    {
        int running = 0;
        const int type_offset = type * MAX_LOCAL_WORLD_SIZE;
        for (int stage = 0; stage < world_size; ++stage)
        {
            const int count = stage_cursors[type_offset + stage];
            stage_cursors[type_offset + stage] = running;
            running += count;
        }
        split_counts[type] = running;
    }
}

__global__ void ag_mt256_scatter_v6_hybrid_tile_lists_kernel(
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ ntiles,
    int *__restrict__ main_tile_info,
    int *__restrict__ tail_tile_info,
    int *__restrict__ stage_cursors,
    int max_tail_rows,
    int world_size,
    int rank,
    bool stage_major,
    int max_tiles)
{
    const int type = blockIdx.x;
    const int stage = threadIdx.x;
    if (type >= 2 || stage >= world_size)
    {
        return;
    }

    constexpr int kTailM = 64;
    int out_problem =
        stage_cursors[type * MAX_LOCAL_WORLD_SIZE + stage];
    const int raw_problem_count = *ntiles;
    const int problem_count =
        raw_problem_count < max_tiles ? raw_problem_count : max_tiles;
    for (int problem = 0; problem < problem_count; ++problem)
    {
        const int *src =
            rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int rows = src[AG_RANK_TILE_ROWS];
        const int problem_stage =
            stage_major
                ? ag_moe_tile_stage(
                      src[AG_RANK_TILE_RANK_END], rank, world_size)
                : 0;
        if (rows <= 0 || problem_stage != stage)
        {
            continue;
        }
        const bool use_tail = rows < 256 && rows <= max_tail_rows;
        if (static_cast<int>(use_tail) != type)
        {
            continue;
        }
        if (type == 0)
        {
            int *dst = main_tile_info +
                       out_problem++ * AG_RANK_TILE_INFO_STRIDE;
#pragma unroll
            for (int field = 0; field < AG_RANK_TILE_INFO_STRIDE; ++field)
            {
                dst[field] = src[field];
            }
            continue;
        }

        const int tail_descriptors = (rows + kTailM - 1) / kTailM;
        for (int tail = 0; tail < tail_descriptors; ++tail)
        {
            const int row_start = tail * kTailM;
            int *dst = tail_tile_info +
                       out_problem++ * AG_RANK_TILE_INFO_STRIDE;
            dst[AG_RANK_TILE_EXPERT] = src[AG_RANK_TILE_EXPERT];
            dst[AG_RANK_TILE_SORTED_START] =
                src[AG_RANK_TILE_SORTED_START] + row_start;
            const int remaining = rows - row_start;
            dst[AG_RANK_TILE_ROWS] =
                remaining < kTailM ? remaining : kTailM;
            dst[AG_RANK_TILE_RANK_START] = src[AG_RANK_TILE_RANK_START];
            dst[AG_RANK_TILE_RANK_END] = src[AG_RANK_TILE_RANK_END];
        }
    }
}

__global__ void ag_mt256_count_v11_early_late_lists_kernel(
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ ntiles,
    int *__restrict__ split_counts,
    int early_stages,
    int prefix_experts,
    int world_size,
    int rank,
    int max_tiles)
{
    const int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem >= max_tiles || problem >= *ntiles)
    {
        return;
    }
    const int *src = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    if (src[AG_RANK_TILE_ROWS] <= 0)
    {
        return;
    }
    const int stage =
        ag_moe_tile_stage(src[AG_RANK_TILE_RANK_END], rank, world_size);
    const bool in_prefix = stage < early_stages &&
                           src[AG_RANK_TILE_EXPERT] < prefix_experts;
    atomicAdd(split_counts + (in_prefix ? 0 : 1), 1);
}

__global__ void ag_mt256_scatter_v11_early_late_lists_kernel(
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ ntiles,
    int *__restrict__ split_tile_info,
    const int *__restrict__ split_counts,
    int early_stages,
    int prefix_experts,
    int world_size,
    int rank,
    int max_tiles)
{
    const int list = blockIdx.x;
    if (list >= 2 || threadIdx.x != 0)
    {
        return;
    }
    int out_problem = list == 0 ? 0 : split_counts[0];
    const int raw_problem_count = *ntiles;
    const int problem_count =
        raw_problem_count < max_tiles ? raw_problem_count : max_tiles;
    for (int problem = 0; problem < problem_count; ++problem)
    {
        const int *src =
            rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        if (src[AG_RANK_TILE_ROWS] <= 0)
        {
            continue;
        }
        const int stage =
            ag_moe_tile_stage(src[AG_RANK_TILE_RANK_END], rank, world_size);
        const bool in_prefix = stage < early_stages &&
                               src[AG_RANK_TILE_EXPERT] < prefix_experts;
        if (static_cast<int>(!in_prefix) != list)
        {
            continue;
        }
        int *dst = split_tile_info +
                   out_problem++ * AG_RANK_TILE_INFO_STRIDE;
#pragma unroll
        for (int field = 0; field < AG_RANK_TILE_INFO_STRIDE; ++field)
        {
            dst[field] = src[field];
        }
    }
}

// Reorder tiles by readiness stage inside small expert windows.  This keeps
// enough expert locality for the weight stream while preventing the first
// wave of CTAs from being dominated by late-rank tiles.
__global__ void ag_mt256_count_v11_window_stage_lists_kernel(
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ ntiles,
    int *__restrict__ bucket_counts,
    int window_experts,
    int world_size,
    int rank,
    int max_tiles)
{
    const int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem >= max_tiles || problem >= *ntiles || window_experts <= 0)
    {
        return;
    }
    const int *src = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    if (src[AG_RANK_TILE_ROWS] <= 0)
    {
        return;
    }
    const int window = src[AG_RANK_TILE_EXPERT] / window_experts;
    const int stage = ag_moe_tile_stage(
        src[AG_RANK_TILE_RANK_END], rank, world_size);
    atomicAdd(bucket_counts + window * world_size + stage, 1);
}

__global__ void ag_mt256_prefix_v11_window_stage_lists_kernel(
    int *__restrict__ bucket_offsets,
    int bucket_count,
    int *__restrict__ split_counts)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    int running = 0;
    for (int bucket = 0; bucket < bucket_count; ++bucket)
    {
        const int count = bucket_offsets[bucket];
        bucket_offsets[bucket] = running;
        running += count;
    }
    split_counts[0] = running;
    split_counts[1] = 0;
}

__global__ void ag_mt256_scatter_v11_window_stage_lists_kernel(
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ ntiles,
    int *__restrict__ stage_tile_info,
    const int *__restrict__ bucket_offsets,
    int window_experts,
    int world_size,
    int rank,
    int max_tiles)
{
    const int bucket = blockIdx.x;
    if (threadIdx.x != 0 || window_experts <= 0)
    {
        return;
    }
    const int window = bucket / world_size;
    const int stage = bucket - window * world_size;
    int out_problem = bucket_offsets[bucket];
    const int problem_count = min(*ntiles, max_tiles);
    for (int problem = 0; problem < problem_count; ++problem)
    {
        const int *src = rank_tile_info +
                         problem * AG_RANK_TILE_INFO_STRIDE;
        if (src[AG_RANK_TILE_ROWS] <= 0 ||
            src[AG_RANK_TILE_EXPERT] / window_experts != window ||
            ag_moe_tile_stage(src[AG_RANK_TILE_RANK_END], rank, world_size) !=
                stage)
        {
            continue;
        }
        int *dst = stage_tile_info +
                   out_problem++ * AG_RANK_TILE_INFO_STRIDE;
#pragma unroll
        for (int field = 0; field < AG_RANK_TILE_INFO_STRIDE; ++field)
        {
            dst[field] = src[field];
        }
    }
}

__device__ inline int ag_mt256_v11_chunk_ready_order(
    const int *problem_info,
    int world_size,
    int rank,
    int chunks_per_rank)
{
    const int chunk_start =
        problem_info[AG_RANK_TILE_PROBLEM_CHUNK_START];
    const int chunk_end = problem_info[AG_RANK_TILE_PROBLEM_CHUNK_END];
    if (chunk_start < 0 || chunk_end < chunk_start || chunks_per_rank <= 0)
    {
        return 0;
    }

    int ready_order = 0;
    const int total_chunks = world_size * chunks_per_rank;
    const int bounded_end = chunk_end < total_chunks
                                ? chunk_end
                                : total_chunks - 1;
    for (int chunk = chunk_start; chunk <= bounded_end; ++chunk)
    {
        const int owner_rank = chunk / chunks_per_rank;
        const int rank_chunk = chunk - owner_rank * chunks_per_rank;
        const int stage =
            (owner_rank - rank + world_size) % world_size;
        // Every local chunk is signaled before GEMM starts. Remote chunks
        // arrive in stage-major, then rank-chunk-major order.
        const int order = stage == 0
                              ? 0
                              : stage * chunks_per_rank + rank_chunk;
        ready_order = order > ready_order ? order : ready_order;
    }
    return ready_order;
}

__global__ void ag_mt256_count_v11_chunk_ready_lists_kernel(
    const int *__restrict__ rank_tile_problem_info,
    const int *__restrict__ ntiles,
    int *__restrict__ ready_counts,
    int world_size,
    int rank,
    int chunks_per_rank,
    int prefix_experts,
    bool stage_major,
    int max_tiles)
{
    const int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem >= max_tiles || problem >= *ntiles)
    {
        return;
    }
    const int *src = rank_tile_problem_info +
                     problem * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
    if (src[AG_RANK_TILE_ROWS] <= 0)
    {
        return;
    }
    const int ready_order = ag_mt256_v11_chunk_ready_order(
        src, world_size, rank, chunks_per_rank);
    const int bootstrap_ready_order = 2 * chunks_per_rank - 1;
    const bool in_prefix = src[AG_RANK_TILE_EXPERT] < prefix_experts &&
                           ready_order <= bootstrap_ready_order;
    const int order = stage_major ? ready_order : (in_prefix ? 0 : 1);
    atomicAdd(ready_counts + order, 1);
}

__global__ void ag_mt256_prefix_v11_chunk_ready_lists_kernel(
    int *__restrict__ ready_offsets,
    int ready_order_count)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    int running = 0;
    for (int order = 0; order < ready_order_count; ++order)
    {
        const int count = ready_offsets[order];
        ready_offsets[order] = running;
        running += count;
    }
}

__global__ void ag_mt256_scatter_v11_chunk_ready_lists_kernel(
    const int *__restrict__ rank_tile_problem_info,
    const int *__restrict__ ntiles,
    int *__restrict__ chunk_tile_info,
    int *__restrict__ chunk_problem_info,
    const int *__restrict__ ready_offsets,
    int world_size,
    int rank,
    int chunks_per_rank,
    int prefix_experts,
    bool stage_major,
    int max_tiles)
{
    const int order = blockIdx.x;
    const int ready_order_count =
        stage_major ? world_size * chunks_per_rank : 2;
    if (order >= ready_order_count || threadIdx.x != 0)
    {
        return;
    }

    int out_problem = ready_offsets[order];
    const int raw_problem_count = *ntiles;
    const int problem_count =
        raw_problem_count < max_tiles ? raw_problem_count : max_tiles;
    for (int problem = 0; problem < problem_count; ++problem)
    {
        const int *src = rank_tile_problem_info +
                         problem * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
        const int ready_order = ag_mt256_v11_chunk_ready_order(
            src, world_size, rank, chunks_per_rank);
        const int bootstrap_ready_order = 2 * chunks_per_rank - 1;
        const bool in_prefix = src[AG_RANK_TILE_EXPERT] < prefix_experts &&
                               ready_order <= bootstrap_ready_order;
        const int problem_order =
            stage_major ? ready_order : (in_prefix ? 0 : 1);
        if (src[AG_RANK_TILE_ROWS] <= 0 || problem_order != order)
        {
            continue;
        }
        int *tile_dst = chunk_tile_info +
                        out_problem * AG_RANK_TILE_INFO_STRIDE;
        int *problem_dst = chunk_problem_info +
                           out_problem * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
#pragma unroll
        for (int field = 0; field < AG_RANK_TILE_INFO_STRIDE; ++field)
        {
            tile_dst[field] = src[field];
        }
#pragma unroll
        for (int field = 0; field < AG_RANK_TILE_PROBLEM_INFO_STRIDE; ++field)
        {
            problem_dst[field] = src[field];
        }
        ++out_problem;
    }
}

__global__ void ag_mt256_build_native_contract_info_kernel(
    const int *__restrict__ rank_tile_info,
    int *__restrict__ contract_tile_info,
    int *__restrict__ sorted_expert_offsets,
    int problem_count,
    int max_experts)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < problem_count)
    {
        const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
        contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
        contract_tile_info[problem_count + idx] = idx * 256;
        contract_tile_info[2 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_START];
        contract_tile_info[3 * problem_count + idx] =
            src[AG_RANK_TILE_RANK_END];
        contract_tile_info[4 * problem_count + idx] =
            src[AG_RANK_TILE_ROWS];
    }
    if (idx <= max_experts)
    {
        sorted_expert_offsets[idx] = 0;
    }
}

__global__ void ag_mt256_build_expert_n_major_task_map_kernel(
    const int *__restrict__ rank_tile_info,
    int32_t *__restrict__ task_map,
    int problem_count,
    int n_tiles,
    int readiness_window_stages)
{
    const int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem >= problem_count)
    {
        return;
    }

    const int expert =
        rank_tile_info[problem * AG_RANK_TILE_INFO_STRIDE + AG_RANK_TILE_EXPERT];
    if (readiness_window_stages > 0)
    {
        const int stage =
            rank_tile_info[problem * AG_RANK_TILE_INFO_STRIDE +
                           AG_RANK_TILE_RANK_END];
        const int readiness_window = stage / readiness_window_stages;
        int group_begin = 0;
        int group_problem_count = 0;
        int local_problem = 0;
        for (int other = 0; other < problem_count; ++other)
        {
            const int *other_info =
                rank_tile_info + other * AG_RANK_TILE_INFO_STRIDE;
            const int other_stage = other_info[AG_RANK_TILE_RANK_END];
            const int other_window =
                other_stage / readiness_window_stages;
            const int other_expert = other_info[AG_RANK_TILE_EXPERT];
            if (other_window < readiness_window ||
                (other_window == readiness_window && other_expert < expert))
            {
                ++group_begin;
            }
            else if (other_window == readiness_window &&
                     other_expert == expert)
            {
                if (other < problem)
                {
                    ++local_problem;
                }
                ++group_problem_count;
            }
        }

        const int group_task_begin = group_begin * n_tiles;
        for (int n_tile = 0; n_tile < n_tiles; ++n_tile)
        {
            const int task = group_task_begin +
                             n_tile * group_problem_count + local_problem;
            task_map[task] = (problem << 6) | n_tile;
        }
        return;
    }

    int expert_begin = problem;
    while (expert_begin > 0 &&
           rank_tile_info[(expert_begin - 1) * AG_RANK_TILE_INFO_STRIDE +
                          AG_RANK_TILE_EXPERT] == expert)
    {
        --expert_begin;
    }
    int expert_end = problem + 1;
    while (expert_end < problem_count &&
           rank_tile_info[expert_end * AG_RANK_TILE_INFO_STRIDE +
                          AG_RANK_TILE_EXPERT] == expert)
    {
        ++expert_end;
    }

    const int expert_problem_count = expert_end - expert_begin;
    const int local_problem = problem - expert_begin;
    const int expert_task_begin = expert_begin * n_tiles;
    for (int n_tile = 0; n_tile < n_tiles; ++n_tile)
    {
        const int task = expert_task_begin +
                         n_tile * expert_problem_count + local_problem;
        task_map[task] = (problem << 6) | n_tile;
    }
}

__global__ void ag_mt256_build_native_direct_input_info_kernel(
    const int *__restrict__ rank_tile_info,
    int *__restrict__ contract_tile_info,
    int problem_count,
    const int *__restrict__ sorted_gather_index,
    int full_m,
    int world_size,
                           bool compact_physical_rows,
                           int current_rank,
                           bool stage_signal_layout,
                           bool ragged_m)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < problem_count)
    {
        const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
        contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
        contract_tile_info[problem_count + idx] =
            src[AG_RANK_TILE_SORTED_START];
        int slot_start = src[AG_RANK_TILE_RANK_START];
        int slot_end = src[AG_RANK_TILE_RANK_END];
        // Native strict-rank problems feed the hand-written V40 body.  Its
        // ready check interprets fields 2/3 as stage slots, while the rank-tile
        // builder stores physical owner ranks.  Convert only for the ragged
        // stage-slot ring so every other ABI remains bit-for-bit unchanged.
        if (ragged_m && stage_signal_layout && slot_start >= 0 &&
            slot_end < world_size && world_size > 0)
        {
            const int owner_start = slot_start;
            const int owner_end = slot_end;
            slot_start =
                (owner_start - current_rank + world_size) % world_size;
            slot_end = (owner_end - current_rank + world_size) % world_size;
            if (slot_start > slot_end)
            {
                slot_end = world_size - 1;
            }
        }
        contract_tile_info[2 * problem_count + idx] = slot_start;
        contract_tile_info[3 * problem_count + idx] = slot_end;
        if (idx < 4 && ragged_m && stage_signal_layout)
        {
            printf("[hygon] rank=%d contract tile=%d owner=%d..%d slot=%d..%d\n",
                   current_rank,
                   idx,
                   src[AG_RANK_TILE_RANK_START],
                   src[AG_RANK_TILE_RANK_END],
                   slot_start,
                   slot_end);
        }
        int rows = src[AG_RANK_TILE_ROWS];
        // For the Megatron routed buffer, rows belonging to one source rank
        // are physically contiguous in the rank-major all-gather result.
        // Encode the first rank boundary in the high half of the row field so
        // the NN body can form addresses without a per-row metadata load.
        // A merged tile spanning more than two ranks keeps split=0 and falls
        // back to the general row-map path.
        if (compact_physical_rows && sorted_gather_index != nullptr &&
            full_m > 0 && world_size > 0 && rows > 0)
        {
            const int local_m = full_m / world_size;
            int split = rows;
            int segments = 1;
            if (local_m > 0)
            {
                int prev_rank = sorted_gather_index[src[AG_RANK_TILE_SORTED_START]] /
                                local_m;
                for (int row = 1; row < rows; ++row)
                {
                    const int token = sorted_gather_index[
                        src[AG_RANK_TILE_SORTED_START] + row];
                    const int owner_rank = token / local_m;
                    if (owner_rank != prev_rank)
                    {
                        if (segments == 1)
                        {
                            split = row;
                        }
                        ++segments;
                        prev_rank = owner_rank;
                    }
                }
            }
            if (segments > 2)
            {
                split = 0;
            }
            contract_tile_info[4 * problem_count + idx] =
                rows | (split << 16);
        }
        else
        {
            contract_tile_info[4 * problem_count + idx] = rows;
        }
    }
}

__global__ void ag_mt256_build_native_direct_input_split_info_kernel(
    const int *__restrict__ rank_tile_info,
    int *__restrict__ contract_tile_info,
    int problem_count,
    int first_problem_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= problem_count)
    {
        return;
    }
    const bool first = idx < first_problem_count;
    const int list_start = first ? 0 : first_problem_count;
    const int list_count = first ? first_problem_count
                                 : problem_count - first_problem_count;
    const int list_problem = idx - list_start;
    int *dst = contract_tile_info + 5 * list_start;
    const int *src = rank_tile_info + idx * AG_RANK_TILE_INFO_STRIDE;
    dst[list_problem] = src[AG_RANK_TILE_EXPERT];
    dst[list_count + list_problem] = src[AG_RANK_TILE_SORTED_START];
    dst[2 * list_count + list_problem] = src[AG_RANK_TILE_RANK_START];
    dst[3 * list_count + list_problem] = src[AG_RANK_TILE_RANK_END];
    dst[4 * list_count + list_problem] = src[AG_RANK_TILE_ROWS];
}

__global__ void ag_mt256_build_native_direct_input_chunk_info_kernel(
    const int *__restrict__ rank_tile_problem_info,
    int *__restrict__ contract_tile_info,
    int problem_count)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= problem_count)
    {
        return;
    }
    const int *src = rank_tile_problem_info +
                     idx * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
    contract_tile_info[idx] = src[AG_RANK_TILE_EXPERT];
    contract_tile_info[problem_count + idx] =
        src[AG_RANK_TILE_SORTED_START];
    contract_tile_info[2 * problem_count + idx] =
        src[AG_RANK_TILE_RANK_START];
    contract_tile_info[3 * problem_count + idx] =
        src[AG_RANK_TILE_RANK_END];
    contract_tile_info[4 * problem_count + idx] = src[AG_RANK_TILE_ROWS];
    contract_tile_info[5 * problem_count + idx] =
        src[AG_RANK_TILE_PROBLEM_CHUNK_START];
    contract_tile_info[6 * problem_count + idx] =
        src[AG_RANK_TILE_PROBLEM_CHUNK_END];
}

__global__ void ag_mt256_build_native_direct_input_rank_chunk_info_kernel(
    const int *__restrict__ rank_tile_problem_info,
    int *__restrict__ contract_tile_info,
    int problem_count,
    int chunks_per_rank,
    int rank,
    int world_size,
    int chunk_slot_base)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= problem_count)
    {
        return;
    }
    const int *src = rank_tile_problem_info +
                     index * AG_RANK_TILE_PROBLEM_INFO_STRIDE;
    int chunk_start = src[AG_RANK_TILE_PROBLEM_CHUNK_START];
    int chunk_end = src[AG_RANK_TILE_PROBLEM_CHUNK_END];
    if (chunk_start < 0 || chunk_end < chunk_start || chunks_per_rank <= 0)
    {
        const int rank_start = src[AG_RANK_TILE_RANK_START];
        const int rank_end = src[AG_RANK_TILE_RANK_END];
        chunk_start = rank_start * chunks_per_rank;
        chunk_end = (rank_end + 1) * chunks_per_rank - 1;
    }
    contract_tile_info[index] = src[AG_RANK_TILE_EXPERT];
    contract_tile_info[problem_count + index] =
        src[AG_RANK_TILE_SORTED_START];
    const int start_owner = chunk_start / chunks_per_rank;
    const int start_chunk = chunk_start - start_owner * chunks_per_rank;
    const int end_owner = chunk_end / chunks_per_rank;
    const int end_chunk = chunk_end - end_owner * chunks_per_rank;
    const int start_stage =
        (start_owner - rank + world_size) % world_size;
    const int end_stage = (end_owner - rank + world_size) % world_size;
    contract_tile_info[2 * problem_count + index] =
        chunk_slot_base + start_stage * chunks_per_rank + start_chunk;
    contract_tile_info[3 * problem_count + index] =
        chunk_slot_base + end_stage * chunks_per_rank + end_chunk;
    contract_tile_info[4 * problem_count + index] = src[AG_RANK_TILE_ROWS];
}

__global__ void ag_mt256_pack_native_rank_tiles_kernel(
    const uint16_t *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    uint16_t *__restrict__ packed_input,
    int problem_count,
    int full_m,
    int k_dim)
{
    const int64_t rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t total = rows * k_dim;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / k_dim);
        const int k = static_cast<int>(idx - static_cast<int64_t>(row) * k_dim);
        const int problem = row / 256;
        const int local_row = row - problem * 256;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        uint16_t value = 0;
        if (local_row < tile_rows)
        {
            const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value = full_input[static_cast<int64_t>(token) * k_dim + k];
            }
        }
        packed_input[idx] = value;
    }
}

__global__ void ag_mt256_pack_native_rank_tiles_tile_wait_kernel(
    const uint16_t *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ rank_signals,
    uint16_t *__restrict__ packed_input,
    int problem_count,
    int full_m,
    int k_dim,
    int world_size,
    int signal_value,
    int k_tile)
{
    const int problem = blockIdx.x;
    const int k_begin = blockIdx.y * k_tile;
    if (problem >= problem_count || k_begin >= k_dim)
    {
        return;
    }
    const int remaining_k = k_dim - k_begin;
    const int tile_k = remaining_k < k_tile ? remaining_k : k_tile;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile[AG_RANK_TILE_RANK_START];
        const int segment_end = tile[AG_RANK_TILE_RANK_END];
        if (segment_start >= 0 && segment_start <= segment_end &&
            segment_end < world_size)
        {
            for (int src_rank = segment_start + threadIdx.x;
                 src_rank <= segment_end;
                 src_rank += blockDim.x)
            {
                ag_wait_signal_ge(rank_signals, src_rank, signal_value);
            }
        }
        __syncthreads();
    }

    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int64_t out_row_base = static_cast<int64_t>(problem) * 256;
    const int tile_elems = 256 * tile_k;
    for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x)
    {
        const int local_row = linear / tile_k;
        const int k = k_begin + (linear - local_row * tile_k);
        uint16_t value = 0;
        if (local_row < tile_rows)
        {
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value =
                    full_input[static_cast<int64_t>(token) * k_dim + k];
            }
        }
        packed_input[(out_row_base + local_row) * k_dim + k] = value;
    }
}

__global__ void ag_mt256_pack_native_rank_tiles_tile_wait_vec8_kernel(
    const uint4 *__restrict__ full_input,
    const int *__restrict__ sorted_gather_index,
    const int *__restrict__ rank_tile_info,
    const int *__restrict__ rank_signals,
    uint4 *__restrict__ packed_input,
    int problem_count,
    int full_m,
    int k_vec_dim,
    int world_size,
    int signal_value,
    int k_tile_vec)
{
    const int problem = blockIdx.x;
    const int k_vec_begin = blockIdx.y * k_tile_vec;
    if (problem >= problem_count || k_vec_begin >= k_vec_dim)
    {
        return;
    }
    const int remaining_k_vec = k_vec_dim - k_vec_begin;
    const int tile_k_vec =
        remaining_k_vec < k_tile_vec ? remaining_k_vec : k_tile_vec;
    const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
    const int tile_rows = tile[AG_RANK_TILE_ROWS];

    if (rank_signals != nullptr)
    {
        const int segment_start = tile[AG_RANK_TILE_RANK_START];
        const int segment_end = tile[AG_RANK_TILE_RANK_END];
        if (segment_start >= 0 && segment_start <= segment_end &&
            segment_end < world_size)
        {
            for (int src_rank = segment_start + threadIdx.x;
                 src_rank <= segment_end;
                 src_rank += blockDim.x)
            {
                ag_wait_signal_ge(rank_signals, src_rank, signal_value);
            }
        }
        __syncthreads();
    }

    const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
    const int64_t out_row_base = static_cast<int64_t>(problem) * 256;
    const int tile_elems = 256 * tile_k_vec;
    for (int linear = threadIdx.x; linear < tile_elems; linear += blockDim.x)
    {
        const int local_row = linear / tile_k_vec;
        const int k_vec = k_vec_begin + (linear - local_row * tile_k_vec);
        uint4 value{0, 0, 0, 0};
        if (local_row < tile_rows)
        {
            const int token = sorted_gather_index[sorted_start + local_row];
            if (token >= 0 && token < full_m)
            {
                value =
                    full_input[static_cast<int64_t>(token) * k_vec_dim +
                               k_vec];
            }
        }
        packed_input[(out_row_base + local_row) * k_vec_dim + k_vec] = value;
    }
}

__global__ void ag_mt256_scatter_native_contract_output_kernel(
    const uint16_t *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint16_t *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_dim)
{
    const int64_t rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t total = rows * n_dim;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / n_dim);
        const int n = static_cast<int>(idx - static_cast<int64_t>(row) * n_dim);
        const int problem = row / 256;
        const int local_row = row - problem * 256;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        if (local_row >= tile_rows)
        {
            continue;
        }
        const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_dim + n] =
                packed_output[idx];
        }
    }
}

__global__ void ag_mt256_scatter_native_contract_output_vec8_kernel(
    const uint4 *__restrict__ packed_output,
    const int *__restrict__ sorted_token_ids,
    const int *__restrict__ rank_tile_info,
    uint4 *__restrict__ output,
    int problem_count,
    int output_rows,
    int n_vec)
{
    const int64_t rows = static_cast<int64_t>(problem_count) * 256;
    const int64_t total = rows * n_vec;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += stride)
    {
        const int row = static_cast<int>(idx / n_vec);
        const int col_vec = static_cast<int>(idx - static_cast<int64_t>(row) * n_vec);
        const int problem = row / 256;
        const int local_row = row - problem * 256;
        const int *tile = rank_tile_info + problem * AG_RANK_TILE_INFO_STRIDE;
        const int tile_rows = tile[AG_RANK_TILE_ROWS];
        if (local_row >= tile_rows)
        {
            continue;
        }
        const int sorted_start = tile[AG_RANK_TILE_SORTED_START];
        const int flat_row = sorted_token_ids[sorted_start + local_row];
        if (flat_row >= 0 && flat_row < output_rows)
        {
            output[static_cast<int64_t>(flat_row) * n_vec + col_vec] =
                packed_output[idx];
        }
    }
}

struct AgMt256Module
{
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    std::string co_path;
    std::string func_name;
};

AgMt256Module &ag_mt256_native_module_cache()
{
    static AgMt256Module cache;
    return cache;
}

std::mutex &ag_mt256_native_module_mutex()
{
    static std::mutex mutex;
    return mutex;
}

constexpr const char *AG_WGRAD_GROUPED_ASM_FILE =
    "Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_"
    "MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4.co";
constexpr const char *AG_WGRAD_GROUPED_ASM_FUNCTION =
    "Custom_Cijk_Ailk_Bjlk_BBS_BH_Bias_GG_AS_SAV_UserArgs_"
    "MT128x256x16_MI16x16x1_SN_GSU1_K1_MIWT4_4";

AgMt256Module &ag_wgrad_grouped_module_cache()
{
    static AgMt256Module cache;
    return cache;
}

std::mutex &ag_wgrad_grouped_module_mutex()
{
    static std::mutex mutex;
    return mutex;
}

hipFunction_t ag_wgrad_get_grouped_asm_function()
{
    const char *explicit_co =
        detail::get_vendor_env("FTRAIN_HYGON_AG_GROUP_GEMM_WGRAD_ASM_CO");
    std::string co_path;
    if (explicit_co != nullptr && explicit_co[0] != '\0')
    {
        co_path = explicit_co;
    }
    else
    {
        const char *asm_dir = detail::get_vendor_env("FTRAIN_HYGON_ASM_DIR_GFX936");
        FTRAIN_HYGON_CHECK(asm_dir != nullptr && asm_dir[0] != '\0')
            << "FC1 wgrad grouped ASM requires FTRAIN_HYGON_ASM_DIR_GFX936 or "
               "FTRAIN_HYGON_AG_GROUP_GEMM_WGRAD_ASM_CO";
        co_path = asm_dir;
        if (!co_path.empty() && co_path.back() != '/')
        {
            co_path.push_back('/');
        }
        co_path += AG_WGRAD_GROUPED_ASM_FILE;
    }
    const char *explicit_function =
        detail::get_vendor_env("FTRAIN_HYGON_AG_GROUP_GEMM_WGRAD_ASM_FUNC");
    const std::string function_name =
        explicit_function != nullptr && explicit_function[0] != '\0'
        ? explicit_function
        : AG_WGRAD_GROUPED_ASM_FUNCTION;

    std::lock_guard<std::mutex> guard(ag_wgrad_grouped_module_mutex());
    AgMt256Module &cache = ag_wgrad_grouped_module_cache();
    if (cache.function != nullptr && cache.co_path == co_path &&
        cache.func_name == function_name)
    {
        return cache.function;
    }
    if (cache.module != nullptr)
    {
        CUDA_CHECK(hipModuleUnload(cache.module));
        cache = AgMt256Module{};
    }

    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    hipError_t error = hipModuleLoad(&module, co_path.c_str());
    FTRAIN_HYGON_CHECK(error == hipSuccess)
        << "FC1 wgrad grouped ASM hipModuleLoad failed for " << co_path
        << ": " << hipGetErrorString(error);
    error = hipModuleGetFunction(
        &function, module, function_name.c_str());
    if (error != hipSuccess)
    {
        hipModuleUnload(module);
        FTRAIN_HYGON_CHECK(false)
            << "FC1 wgrad grouped ASM hipModuleGetFunction failed for "
            << function_name << " in " << co_path << ": "
            << hipGetErrorString(error);
    }
    cache.module = module;
    cache.function = function;
    cache.co_path = co_path;
    cache.func_name = function_name;
    return cache.function;
}

struct __attribute__((packed)) AgWgradGroupedAsmLaunchArgs
{
    uint32_t grouped_sizes;
    void *device_arguments;
    void *args_ptr;
    uint32_t wg_table;
    uint32_t gsu;
    uint32_t reserved;
};

static_assert(sizeof(AgWgradGroupedAsmLaunchArgs) == 32,
              "FC1 wgrad grouped ASM kernarg must stay 32 bytes");

void ag_wgrad_launch_grouped_asm(void *grouped_asm_args,
                                 int output_k_extent,
                                 int reduction_dim,
                                 int num_experts,
                                 hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(grouped_asm_args != nullptr);
    FTRAIN_HYGON_CHECK(output_k_extent > 0 && reduction_dim > 0 && num_experts > 0);
    hipFunction_t function = ag_wgrad_get_grouped_asm_function();
    // Keep the in-tree MT128x256 kernel as the production default, but allow
    // an ABI-compatible hipBLASLt grouped UserArgs kernel to be evaluated
    // without changing the wgrad descriptor or metadata path.  In particular,
    // DTK 26.04 provides MT256x256 candidates with the same 196-byte per-
    // expert descriptor.  The grid must match the selected macro tile: a
    // mismatched grid silently skips output tiles or launches duplicate work.
    const int tile_i = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_WGRAD_ASM_TILE_I", 128);
    const int tile_j = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_WGRAD_ASM_TILE_J", 256);
    FTRAIN_HYGON_CHECK((tile_i == 128 || tile_i == 256) && tile_j == 256)
        << "FC1 wgrad grouped ASM currently supports macro tiles "
           "128x256 and 256x256, got "
        << tile_i << "x" << tile_j;
    const uint64_t tiles_i =
        (static_cast<uint64_t>(reduction_dim) + tile_i - 1u) / tile_i;
    const uint64_t tiles_j =
        (static_cast<uint64_t>(output_k_extent) + tile_j - 1u) / tile_j;
    const uint64_t workgroups =
        static_cast<uint64_t>(num_experts) * tiles_i * tiles_j;
    const uint64_t global_threads = workgroups * 512u;
    FTRAIN_HYGON_CHECK(global_threads <= UINT32_MAX)
        << "FC1 wgrad grouped ASM grid exceeds the module launch limit: "
        << global_threads;

    AgWgradGroupedAsmLaunchArgs args{};
    args.grouped_sizes = static_cast<uint32_t>(num_experts);
    args.device_arguments = grouped_asm_args;
    args.args_ptr = nullptr;
    args.wg_table = 0u;
    args.gsu = 1u;
    args.reserved = 0u;
    size_t arg_size = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size,
                      HIP_LAUNCH_PARAM_END};
    CUDA_CHECK(hipExtModuleLaunchKernel(function,
                                        static_cast<uint32_t>(global_threads),
                                        1,
                                        1,
                                        512,
                                        1,
                                        1,
                                        0,
                                        stream,
                                        nullptr,
                                        config,
                                        nullptr,
                                        nullptr,
                                        0));
}

struct AgMt256NativeProfile
{
    std::array<hipEvent_t, 7> events{};
    bool active = false;

    bool init(bool enable)
    {
        if (!enable)
        {
            return false;
        }
        int created = 0;
        for (auto &event : events)
        {
            if (hipEventCreate(&event) != hipSuccess)
            {
                for (int i = 0; i < created; ++i)
                {
                    hipEventDestroy(events[i]);
                    events[i] = nullptr;
                }
                return false;
            }
            ++created;
        }
        active = true;
        return true;
    }

    void record(int index, hipStream_t stream)
    {
        if (active)
        {
            hipEventRecord(events[index], stream);
        }
    }

    void print(int rank, int call_index)
    {
        if (!active)
        {
            return;
        }
        hipEventSynchronize(events[6]);
        std::array<float, 6> stage_ms{};
        for (int i = 0; i < 6; ++i)
        {
            hipEventElapsedTime(&stage_ms[i], events[i], events[i + 1]);
        }
        float total_ms = 0.0f;
        hipEventElapsedTime(&total_ms, events[0], events[6]);
        printf("[hygon] ASM MT256 profile rank=%d iter=%d "
               "problem_count_ms=%.3f info_ms=%.3f module_setup_ms=%.3f "
               "input_pack_ms=%.3f gemm_ms=%.3f scatter_ms=%.3f "
               "total_ms=%.3f\n",
               rank,
               call_index,
               stage_ms[0],
               stage_ms[1],
               stage_ms[2],
               stage_ms[3],
               stage_ms[4],
               stage_ms[5],
               total_ms);
        fflush(stdout);
    }

    ~AgMt256NativeProfile()
    {
        for (auto &event : events)
        {
            if (event != nullptr)
            {
                hipEventDestroy(event);
            }
        }
    }
};

std::string canonical_rank_tile_function_name(const char *name)
{
    if (name == nullptr)
    {
        return {};
    }
    const std::string value(name);
    if (value ==
        "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_"
        "stage_ordered_ready_queue_ag_frontload16_win64_tailmerge_wgm8")
    {
        return "hygon_ag_grouped_gemm_rank_tile_dynamic_wgm8";
    }
    if (value ==
        "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_stage_ordered_"
        "ready_queue_ag_frontload16_win64_tailmerge_wgm8")
    {
        return "hygon_ag_grouped_gemm_rank_tile_fixed_n8_wgm8";
    }
    if (value ==
        "hygon_ag_grouped_gemm_v36_direct_flat_stage_expert_nmajor_wgm1")
    {
        return "hygon_ag_grouped_gemm_rank_tile_expert_nmajor_wgm1";
    }
    if (value ==
        "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_"
        "direct_output_nchunk_signal")
    {
        return "hygon_ag_grouped_gemm_rank_tile_dynamic_direct_wgm8";
    }
    if (value ==
        "hygon_ag_grouped_gemm_v36_direct_flat_stage_expert_nmajor_wgm1_"
        "direct_output_global64")
    {
        return "hygon_ag_grouped_gemm_rank_tile_expert_nmajor_direct_wgm1";
    }
    return value;
}

std::string legacy_rank_tile_function_name(const std::string &name)
{
    if (name == "hygon_ag_grouped_gemm_rank_tile_dynamic_wgm8")
    {
        return "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_"
               "stage_ordered_ready_queue_ag_frontload16_win64_tailmerge_"
               "wgm8";
    }
    if (name == "hygon_ag_grouped_gemm_rank_tile_fixed_n8_wgm8")
    {
        return "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_stage_"
               "ordered_ready_queue_ag_frontload16_win64_tailmerge_wgm8";
    }
    if (name == "hygon_ag_grouped_gemm_rank_tile_expert_nmajor_wgm1")
    {
        return "hygon_ag_grouped_gemm_v36_direct_flat_stage_expert_nmajor_"
               "wgm1";
    }
    if (name == "hygon_ag_grouped_gemm_rank_tile_dynamic_direct_wgm8")
    {
        return "hygon_ag_grouped_gemm_contract_rank_tile_x64_v36_dynamic_"
               "direct_output_nchunk_signal";
    }
    if (name ==
        "hygon_ag_grouped_gemm_rank_tile_expert_nmajor_direct_wgm1")
    {
        return "hygon_ag_grouped_gemm_v36_direct_flat_stage_expert_nmajor_"
               "wgm1_direct_output_global64";
    }
    return name;
}

bool ag_mt256_get_native_contract_function(
    hipFunction_t *function,
    bool force_v5_wgm8_fallback,
    bool use_v36_dynamic,
    bool use_v36_fixed_n8,
    bool use_v36_expert_n_major,
    bool use_v36_expert_n_major_direct_output,
    bool ragged_m)
{
    const bool direct_input = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false);
    const bool wgm8 = direct_input && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8", false);
    const bool v36_direct_output = wgm8 && use_v36_dynamic &&
        !force_v5_wgm8_fallback && get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_DIRECT_OUTPUT",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_DIRECT_OUTPUT",
            false);
    const bool direct_output = wgm8 &&
        (v36_direct_output || get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", false));
    const bool v6_signal_gate = wgm8 && !direct_output && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V6", false);
    const bool v11_chunk_signal = wgm8 && !direct_output &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11", false) &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_CHUNK_SIGNAL", false);
    const bool v36_stage_ordered = wgm8 &&
        (!direct_output || v36_direct_output) &&
        !force_v5_wgm8_fallback && get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_ENABLE",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_STAGE_ORDERED",
            false);
    const bool v36_dynamic = v36_stage_ordered && use_v36_dynamic;
    const bool v40_true_ring = v36_stage_ordered && !v36_dynamic &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_V40_TRUE_RING", false);
    const bool v12_ready_queue = wgm8 && !direct_output &&
        (v36_stage_ordered ||
         get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V12_READY_QUEUE", false));
    const bool oob_store = direct_output && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_OOB_STORE", false);
    const bool lds_rowmap = direct_output && !oob_store && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_LDS_ROWMAP", false);
    const char *default_co =
        use_v36_expert_n_major_direct_output
            ? "hygon_ag_grouped_gemm_rank_tile_expert_nmajor_direct_wgm1.co"
        : v36_direct_output
            ? "hygon_ag_grouped_gemm_rank_tile_dynamic_direct_wgm8.co"
        : oob_store
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v10_direct_output_oob_store_wgm8.co"
        : lds_rowmap
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v9_direct_output_lds_rowmap_wgm8.co"
        : direct_output
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v8_direct_output_wgm8.co"
        : v6_signal_gate
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v6_signal_gate_wgm8.co"
        : v40_true_ring
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v40_ring_global_ready_queue_wgm8.co"
        : v36_dynamic
            ? "hygon_ag_grouped_gemm_rank_tile_dynamic_bundle.co"
        : v36_stage_ordered
            ? "hygon_ag_grouped_gemm_rank_tile_fixed_n8_wgm8.co"
        : v12_ready_queue
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v12_ready_queue_wgm8.co"
        : v11_chunk_signal
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v11_chunk_signal_wgm8.co"
        : wgm8
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v5_direct_input_wgm8.co"
        : direct_input
            ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v4_direct_input.co"
            : "hygon_ag_grouped_gemm_contract_rank_tile_x64_v1.co";
    const char *co_env = force_v5_wgm8_fallback
        ? nullptr
        : (v36_stage_ordered
               ? get_env_with_legacy(
                     "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_CO",
                     "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_CO")
               : detail::get_vendor_env(
                     "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_CO"));
    std::string co_path;
    if (co_env != nullptr && *co_env != '\0')
    {
        co_path = co_env;
    }
    else
    {
        const char *asm_dir = detail::get_vendor_env("FTRAIN_HYGON_ASM_DIR_GFX936");
        if (asm_dir == nullptr || *asm_dir == '\0')
        {
            return false;
        }
        co_path = asm_dir;
        if (!co_path.empty() && co_path.back() != '/')
        {
            co_path.push_back('/');
        }
        co_path += default_co;
    }
    if (co_path.empty())
    {
        return false;
    }
    const char *func_env = force_v5_wgm8_fallback
        ? nullptr
        : (v36_stage_ordered
               ? get_env_with_legacy(
                     "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_FUNC",
                     "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_FUNC")
               : detail::get_vendor_env(
                     "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_FUNC"));
    const char *expert_func_env = force_v5_wgm8_fallback
        ? nullptr
        : get_env_with_legacy(
              "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_EXPERT_N_MAJOR_FUNC",
              "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_EXPERT_N_MAJOR_FUNC");
    constexpr const char *kRankTileExpertNMajorFunction =
        "hygon_ag_grouped_gemm_rank_tile_expert_nmajor_wgm1";
    constexpr const char *kRankTileDirectOutputFunction =
        "hygon_ag_grouped_gemm_rank_tile_dynamic_direct_wgm8";
    constexpr const char *kRankTileExpertNMajorDirectOutputFunction =
        "hygon_ag_grouped_gemm_rank_tile_expert_nmajor_direct_wgm1";
    constexpr const char *kRankTileFixedN8Function =
        "hygon_ag_grouped_gemm_rank_tile_fixed_n8_wgm8";
    std::string func_name =
        use_v36_expert_n_major_direct_output && func_env != nullptr &&
                *func_env != '\0'
            ? canonical_rank_tile_function_name(func_env)
        : use_v36_expert_n_major_direct_output
            ? std::string(kRankTileExpertNMajorDirectOutputFunction)
        : v36_direct_output && func_env != nullptr && *func_env != '\0'
            ? canonical_rank_tile_function_name(func_env)
        : v36_direct_output
            ? std::string(kRankTileDirectOutputFunction)
        : use_v36_expert_n_major && expert_func_env != nullptr &&
                *expert_func_env != '\0'
            ? canonical_rank_tile_function_name(expert_func_env)
        : use_v36_expert_n_major && func_env != nullptr &&
                std::strstr(func_env, "expert_nmajor") != nullptr
            ? canonical_rank_tile_function_name(func_env)
        : use_v36_expert_n_major
            ? std::string(kRankTileExpertNMajorFunction)
        : (func_env != nullptr && *func_env != '\0')
            ? canonical_rank_tile_function_name(func_env)
        : use_v36_fixed_n8
            ? std::string(kRankTileFixedN8Function)
            : std::string(
                  oob_store
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v10_direct_output_oob_store_wgm8"
                  : lds_rowmap
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v9_direct_output_lds_rowmap_wgm8"
                  : direct_output
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v8_direct_output_wgm8"
                  : v6_signal_gate
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v6_signal_gate_wgm8"
                  : v40_true_ring
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v40_ring_global_ready_queue_wgm8"
                  : v36_dynamic
                      ? "hygon_ag_grouped_gemm_rank_tile_dynamic_wgm8"
                  : v36_stage_ordered
                      ? "hygon_ag_grouped_gemm_rank_tile_fixed_n8_wgm8"
                  : v12_ready_queue
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v12_ready_queue_wgm8"
                  : v11_chunk_signal
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v11_chunk_signal_wgm8"
                  : wgm8
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v5_direct_input_wgm8"
                  : direct_input
                      ? "hygon_ag_grouped_gemm_contract_rank_tile_x64_v4_direct_input"
                      : "hygon_ag_grouped_gemm_contract_rank_tile_x64_v1");
    func_name = detail::legacy_kernel_function_name(func_name);
    std::lock_guard<std::mutex> guard(ag_mt256_native_module_mutex());
    AgMt256Module &cache = ag_mt256_native_module_cache();
    if (cache.function != nullptr && cache.co_path == co_path &&
        cache.func_name == func_name)
    {
        *function = cache.function;
        return true;
    }
    if (cache.module != nullptr)
    {
        hipModuleUnload(cache.module);
        cache = AgMt256Module{};
    }
    hipModule_t module = nullptr;
    hipFunction_t loaded_function = nullptr;
    hipError_t err = hipModuleLoad(&module, co_path.c_str());
    if (err != hipSuccess)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            fprintf(stderr,
                    "[hygon] ASM MT256 hipModuleLoad failed for %s: %s\n",
                    co_path.c_str(),
                    hipGetErrorString(err));
        }
        return false;
    }
    err = hipModuleGetFunction(&loaded_function, module, func_name.c_str());
    const std::string legacy_func_name =
        legacy_rank_tile_function_name(func_name);
    if (err != hipSuccess && legacy_func_name != func_name)
    {
        err = hipModuleGetFunction(
            &loaded_function, module, legacy_func_name.c_str());
    }
    const std::string prefix_legacy_func_name =
        detail::legacy_kernel_function_name(func_name);
    if (err != hipSuccess && prefix_legacy_func_name != func_name)
    {
        err = hipModuleGetFunction(
            &loaded_function, module, prefix_legacy_func_name.c_str());
    }
    if (err != hipSuccess)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            fprintf(stderr,
                    "[hygon] ASM MT256 hipModuleGetFunction failed for %s in "
                    "%s: %s\n",
                    func_name.c_str(),
                    co_path.c_str(),
                    hipGetErrorString(err));
        }
        hipModuleUnload(module);
        return false;
    }
    cache.module = module;
    cache.function = loaded_function;
    cache.co_path = co_path;
    cache.func_name = func_name;
    *function = loaded_function;
    return true;
}

bool ag_mt256_launch_native_contract(
    const void *full_input,
    const void *weight,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *rank_tile_info,
    const int *rank_tile_problem_info,
    const int *ntiles,
    const int *rank_signals,
    int *n_chunk_done_counts,
    int *n_chunk_done_flags,
    void *output,
    int full_m,
    int n_dim,
    int k_dim,
    int topk,
    int num_experts,
    int world_size,
    int rank,
    int max_tiles,
    bool weight_nk,
    bool metadata_cache_hit,
    bool ragged_m,
    int signal_value,
    int signal_chunk_rows,
    int chunks_per_rank,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    if (!get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_NATIVE_CONTRACT",
                            false))
    {
        return false;
    }
    const bool v36_direct_output_env = get_env_bool_with_legacy(
        "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_DIRECT_OUTPUT",
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_DIRECT_OUTPUT",
        false);
    const bool v36_dynamic_topk =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false) &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8", false) &&
        (!get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", false) ||
         v36_direct_output_env) &&
        get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_ENABLE",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_STAGE_ORDERED",
            false);
    if (dtype != c10::ScalarType::BFloat16 || !weight_nk || topk <= 0 ||
        (topk != 6 && !v36_dynamic_topk) ||
        full_input == nullptr || weight == nullptr ||
        sorted_token_ids == nullptr || sorted_gather_index == nullptr ||
        rank_tile_info == nullptr || ntiles == nullptr ||
        rank_signals == nullptr || output == nullptr ||
        n_dim <= 0 || k_dim <= 0 || full_m <= 0)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 native contract skipped: requires BF16, "
                   "weight_nk=true, rank signals and rank-tile metadata; "
                   "dynamic topk is supported by V36 only\n");
        }
        return false;
    }

    const bool tile_wait_pack = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TILE_WAIT_PACK", false);
    const bool direct_input = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_INPUT", false);
    const bool compact_physical_rows = direct_input && weight_nk &&
        !ragged_m && get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_NN_COMPACT_PHYSICAL_ROWS", false);
    const bool wgm8 = direct_input && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8", false);
    const bool allow_wide_n = wgm8 && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8_ALLOW_WIDE_N", false);
    const bool direct_output_requested = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT", false);
    const int n_tiles = (n_dim + 255) / 256;
    const bool v36_direct_output_requested =
        v36_direct_output_env && get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_ENABLE",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_STAGE_ORDERED",
            false) &&
        get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_DYNAMIC",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_DYNAMIC",
            true);
    const uint64_t direct_output_bytes =
        static_cast<uint64_t>(full_m) * static_cast<uint64_t>(topk) *
        static_cast<uint64_t>(n_dim) * sizeof(uint16_t);
    // Both direct epilogues use 32-bit buffer offsets.  Keep the per-rank
    // final-output allocation below 4 GiB so the expert-direct route cannot
    // silently wrap its row-mapped stores.
    const bool v36_direct_output = wgm8 &&
        v36_direct_output_requested &&
        direct_output_bytes <= 0xfffffffeULL;
    const bool direct_output = wgm8 &&
        (direct_output_requested || v36_direct_output);
    const bool v36_stage_ordered_requested =
        wgm8 && (!direct_output || v36_direct_output) &&
        get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_ENABLE",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_STAGE_ORDERED",
            false);
    const bool v40_true_ring_requested = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_V40_TRUE_RING", false);
    const bool v36_dynamic = v36_stage_ordered_requested &&
        (!v40_true_ring_requested || ragged_m) &&
        n_tiles > 0 && n_tiles <= 64 && get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_DYNAMIC",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_DYNAMIC",
            true);
    const bool benchmark_preloaded_input = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_BENCHMARK_PRELOADED_INPUT", false);
    const char *v36_co_override = get_env_with_legacy(
        "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_CO",
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_CO");
    const bool v36_dual_kernel_default =
        v36_co_override == nullptr ||
        std::strstr(v36_co_override, "rank_tile_dynamic_bundle") != nullptr ||
        (std::strstr(v36_co_override, "dual") != nullptr &&
         std::strstr(v36_co_override, "expert_nmajor") != nullptr);
    const bool v36_dual_kernel = get_env_bool_with_legacy(
        "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_BUNDLE",
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_DUAL_KERNEL",
        v36_dual_kernel_default);
    // Expert-N-major improves weight-tile locality only when there are enough
    // M-tile problems to amortize its flat task-map/WGM1 scheduling.  EP and
    // topk are carried by the dynamic metadata/task map and are not kernel
    // specialization constraints, so keep the automatic decision independent
    // of their particular values.
    const bool v36_expert_n_major_tuned_shape =
        full_m >= 32768 &&
        ((n_tiles == 7 && k_dim >= 4096 && k_dim <= 32768) ||
         (n_tiles == 8 && k_dim >= 8192 && k_dim <= 32768) ||
         (n_tiles == 16 && k_dim >= 4096) ||
         (n_tiles == 32 && k_dim >= 4096 && k_dim <= 16384) ||
         (n_tiles == 64 && k_dim >= 4096 && k_dim <= 8192));
    const bool v36_expert_n_major_default = v36_dynamic &&
        !v36_direct_output &&
        v36_dual_kernel && v36_expert_n_major_tuned_shape;
    // Direct output uses the same tuned-shape expert-N-major default as the
    // packed path.  An explicit environment value can still disable it.
    const bool v36_expert_n_major_direct_output =
        v36_dynamic && v36_direct_output && get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_EXPERT_N_MAJOR_DIRECT_OUTPUT",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_EXPERT_N_MAJOR_DIRECT_OUTPUT",
            v36_expert_n_major_tuned_shape);
    const bool v36_expert_n_major = v36_dynamic &&
        (v36_expert_n_major_direct_output ||
         (!v36_direct_output && get_env_bool_with_legacy(
              "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_EXPERT_N_MAJOR",
              "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_EXPERT_N_MAJOR",
              v36_expert_n_major_default)));
    const char *v36_func_override = get_env_with_legacy(
        "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_FUNC",
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_FUNC");
    const bool v36_fixed_n8_eligible = v36_dynamic && !v36_direct_output &&
        v36_dual_kernel &&
        !v36_expert_n_major && world_size == 8 && full_m == 16384 &&
        k_dim == 16384 && n_tiles == 8 && num_experts >= 64 &&
        (v36_func_override == nullptr || *v36_func_override == '\0');
    // Keep the fixed-N8 object for historical compatibility, but do not
    // dispatch it for any new route. V36 uses ordinary dynamic scheduling
    // (or expert-N-major when its tuned-shape predicate matches).
    const bool v36_fixed_n8 = false;
    const bool v36_flat_grid = v36_dynamic &&
        (v36_expert_n_major_direct_output ||
         (!v36_direct_output &&
          (v36_expert_n_major || get_env_bool_with_legacy(
               "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_FLAT_GRID",
               "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_FLAT_GRID",
               false))));
    const bool v36_expert_n_major_stage_major = v36_expert_n_major &&
        get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_EXPERT_N_MAJOR_STAGE_MAJOR",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_EXPERT_N_MAJOR_STAGE_MAJOR",
            !benchmark_preloaded_input);
    int v36_expert_n_major_readiness_window =
        v36_expert_n_major_stage_major
            ? get_env_int_with_legacy(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_EXPERT_N_MAJOR_"
                  "READINESS_WINDOW_STAGES",
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_EXPERT_N_MAJOR_"
                  "READINESS_WINDOW_STAGES",
                  world_size == 8 && full_m >= 32768 ? 2 : 1)
            : 0;
    v36_expert_n_major_readiness_window =
        v36_expert_n_major_stage_major
            ? std::max(
                  1,
                  std::min(v36_expert_n_major_readiness_window, world_size))
            : 0;
    const bool v36_stage_ordered =
        v36_stage_ordered_requested && (n_tiles == 8 || v36_dynamic);
    const bool v12_ready_queue_requested =
        wgm8 && (!direct_output || v36_direct_output) &&
        (v36_stage_ordered ||
         get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V12_READY_QUEUE", false));
    const bool v11_enabled = wgm8 && !direct_output && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11", false);
    const bool v11_chunk_signal =
        v11_enabled && rank_tile_problem_info != nullptr &&
        signal_chunk_rows > 0 && chunks_per_rank > 0 &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_CHUNK_SIGNAL", false);
    // The ragged native-KN body consumes the V40 stage-slot ring by default.
    // The chunk-ready ABI is an explicit opt-in and must mirror the transport
    // selection in ag_group_gemm.cpp, otherwise the body waits on chunk slots
    // that the host never publishes.
    const bool ragged_ring_chunk = ragged_m &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_V40_TRUE_RING", false) &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_RING_CHUNK", false);
    const bool v36_rank_chunk_signal =
        direct_input && rank_tile_problem_info != nullptr &&
        signal_chunk_rows > 0 && chunks_per_rank > 0 &&
        (get_env_bool_local(
             "FTRAIN_HYGON_AG_GROUP_GEMM_V40_PARTIAL_READY_RANK_CHUNKS", false) ||
         ragged_ring_chunk);
    if (get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_DEBUG_RANK_CHUNK_INFO", false))
    {
        printf("[hygon] rank-chunk state rank=%d direct=%d problem_info=%p "
               "rows=%d chunks=%d enabled=%d\n",
               rank,
               direct_input ? 1 : 0,
               rank_tile_problem_info,
               signal_chunk_rows,
               chunks_per_rank,
               v36_rank_chunk_signal ? 1 : 0);
        fflush(stdout);
    }
    const bool v11_chunk_stage_major = v11_chunk_signal &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_CHUNK_STAGE_MAJOR", false);
    const bool v11_stage_major = v11_enabled && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_STAGE_MAJOR", false) &&
        !v11_chunk_signal;
    const bool v11_window_stage_major =
        v11_enabled && !v11_chunk_signal && !v11_stage_major &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_WINDOW_STAGE_MAJOR", false);
    const bool v11_split_launch = v11_enabled && !v11_stage_major &&
        !v11_window_stage_major &&
        !v11_chunk_signal &&
        get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_SPLIT_LAUNCH", false);
    const bool v11_ready_prefix = v11_enabled && !v11_chunk_signal &&
        !v11_stage_major && !v11_window_stage_major && !v11_split_launch;
    int v11_early_stages = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_EARLY_STAGES", 1);
    v11_early_stages =
        std::max(1, std::min(v11_early_stages, world_size - 1));
    int v11_prefix_experts = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_PREFIX_EXPERTS", 8);
    v11_prefix_experts = std::max(1, std::min(v11_prefix_experts, 256));
    int v11_window_experts = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_WINDOW_EXPERTS", 8);
    v11_window_experts = std::max(1, std::min(v11_window_experts, 256));
    const bool v11_chunk_ready_prefix = v11_chunk_signal &&
        !v11_chunk_stage_major && get_env_bool_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V11_CHUNK_READY_PREFIX", false);
    const bool v11_chunk_reorder =
        v11_chunk_stage_major || v11_chunk_ready_prefix;
    const bool v6_signal_gate = wgm8 && !direct_output && !v11_enabled &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V6", false);
    const bool v6_hybrid_tail = v6_signal_gate && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V6_HYBRID_TAIL", false);
    const bool v6_tail_vec8 = v6_hybrid_tail && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V6_TAIL_VEC8", false);
    const bool v6_tile_wait = v6_signal_gate && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V6_TILE_WAIT", true);
    const bool v6_stage_major = v6_hybrid_tail && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V6_STAGE_MAJOR", false);
    const bool oob_store = direct_output && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_OOB_STORE", false);
    const bool lds_rowmap = direct_output && !oob_store && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_DIRECT_OUTPUT_LDS_ROWMAP", false);
    if ((direct_output_requested || v36_direct_output_requested) &&
        !direct_output && !v36_direct_output_requested)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 direct-output skipped: requires "
                   "direct-input and WGM8\n");
        }
        return false;
    }
    if (v36_direct_output_requested && !v36_direct_output)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (rank == 0 && logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] V36 direct-output disabled: output span %llu "
                   "bytes exceeds the 32-bit buffer-offset contract; "
                   "using packed output\n",
                   static_cast<unsigned long long>(direct_output_bytes));
        }
    }
    if (direct_input && (k_dim % 64 != 0))
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 direct-input skipped: k_dim must be "
                   "a multiple of 64\n");
        }
        return false;
    }
    // V36 dynamic supports up to 64 N tiles; regular V5/V12 dispatch retains
    // its original eight-tile default unless wide-N is requested.
    const int default_wgm8_n_tiles = v36_dynamic ? 64 : 8;
    int max_wgm8_n_tiles =
        allow_wide_n
            ? get_env_int_local(
                  "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_WGM8_MAX_N_TILES", 64)
            : default_wgm8_n_tiles;
    max_wgm8_n_tiles = std::max(8, std::min(max_wgm8_n_tiles, 64));
    if (wgm8 && n_tiles > max_wgm8_n_tiles)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 WGM8 skipped: source variant supports "
                   "at most %d N tiles in this mode\n",
                   max_wgm8_n_tiles);
        }
        return false;
    }
    const bool v12_ready_queue = v12_ready_queue_requested &&
        !v11_enabled && !v6_signal_gate &&
        (n_tiles == 8 || v36_dynamic);
    if (v12_ready_queue_requested && !v12_ready_queue)
    {
        static std::atomic<bool> logged{false};
        bool expected = false;
        if (logged.compare_exchange_strong(expected, true))
        {
            printf("[hygon] ASM MT256 V12 ready queue skipped: requires "
                   "the plain WGM8 path with exactly 8 N tiles\n");
        }
        return false;
    }
    const bool profile_requested = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_PROFILE", false);
    static std::atomic<int> profile_call_count{0};
    const int profile_call_index =
        profile_requested ? profile_call_count.fetch_add(1) : -1;
    const int requested_profile_skip = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_PROFILE_SKIP", 0);
    const int profile_skip =
        requested_profile_skip > 0 ? requested_profile_skip : 0;
    const int profile_limit = get_env_int_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_PROFILE_ITERS", 4);
    AgMt256NativeProfile profile;
    profile.init(profile_requested && profile_call_index >= profile_skip &&
                 profile_call_index < profile_skip + profile_limit);
    profile.record(0, stream);

    hipError_t err = hipSuccess;
    if (!tile_wait_pack && !direct_input)
    {
        hipLaunchKernelGGL(wait_rank_signals_system_kernel,
                           dim3(1),
                           dim3(1),
                           0,
                           stream,
                           rank_signals,
                           world_size,
                           signal_value);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }
    }

    constexpr int kMaxExpertsForContract = 256;
    int original_problem_count = 0;
    int problem_count = 0;
    int tail_problem_count = 0;
    int v11_first_problem_count = 0;
    const int *active_rank_tile_info = rank_tile_info;
    const int *active_rank_tile_problem_info = rank_tile_problem_info;
    void *v6_main_tile_info = nullptr;
    void *v6_tail_tile_info = nullptr;
    void *v6_split_counts = nullptr;
    void *v6_stage_cursors = nullptr;
    void *v11_stage_tile_info = nullptr;
    void *v11_split_counts = nullptr;
    void *v11_stage_cursors = nullptr;
    void *v11_window_offsets = nullptr;
    void *v11_chunk_tile_info = nullptr;
    void *v11_chunk_problem_info = nullptr;
    void *v11_chunk_ready_offsets = nullptr;
    if (v11_chunk_reorder)
    {
        const size_t max_problem_capacity = std::max(max_tiles, 1);
        const size_t tile_info_bytes = max_problem_capacity *
                                       AG_RANK_TILE_INFO_STRIDE * sizeof(int);
        const size_t problem_info_bytes = max_problem_capacity *
            AG_RANK_TILE_PROBLEM_INFO_STRIDE * sizeof(int);
        const int ready_order_count = v11_chunk_stage_major
                                          ? world_size * chunks_per_rank
                                          : 2;
        if (!ag_mt256_ensure_scratch(&g_mt256_v11_chunk_tile_info,
                                     tile_info_bytes,
                                     &v11_chunk_tile_info) ||
            !ag_mt256_ensure_scratch(&g_mt256_v11_chunk_problem_info,
                                     problem_info_bytes,
                                     &v11_chunk_problem_info) ||
            !ag_mt256_ensure_scratch(&g_mt256_v11_chunk_ready_offsets,
                                     ready_order_count * sizeof(int),
                                     &v11_chunk_ready_offsets))
        {
            return false;
        }

        AgMt256V11ChunkCache &cache = g_mt256_v11_chunk_cache;
        const bool chunk_cache_hit =
            metadata_cache_hit && cache.valid &&
            cache.rank_tile_problem_info == rank_tile_problem_info &&
            cache.ntiles == ntiles &&
            cache.chunk_tile_info == v11_chunk_tile_info &&
            cache.chunk_problem_info == v11_chunk_problem_info &&
            cache.max_tiles == max_tiles &&
            cache.world_size == world_size && cache.rank == rank &&
            cache.chunks_per_rank == chunks_per_rank &&
            cache.prefix_experts == v11_prefix_experts &&
            cache.stage_major == v11_chunk_stage_major;
        if (chunk_cache_hit)
        {
            problem_count = cache.problem_count;
        }
        else
        {
            cache.valid = false;
            err = hipMemsetAsync(v11_chunk_ready_offsets,
                                 0,
                                 ready_order_count * sizeof(int),
                                 stream);
            if (err != hipSuccess)
            {
                return false;
            }
            constexpr int reorder_threads = 256;
            const int reorder_blocks =
                std::max((max_tiles + reorder_threads - 1) /
                             reorder_threads,
                         1);
            hipLaunchKernelGGL(
                ag_mt256_count_v11_chunk_ready_lists_kernel,
                dim3(reorder_blocks),
                dim3(reorder_threads),
                0,
                stream,
                rank_tile_problem_info,
                ntiles,
                static_cast<int *>(v11_chunk_ready_offsets),
                world_size,
                rank,
                chunks_per_rank,
                v11_prefix_experts,
                v11_chunk_stage_major,
                max_tiles);
            hipLaunchKernelGGL(
                ag_mt256_prefix_v11_chunk_ready_lists_kernel,
                dim3(1),
                dim3(1),
                0,
                stream,
                static_cast<int *>(v11_chunk_ready_offsets),
                ready_order_count);
            hipLaunchKernelGGL(
                ag_mt256_scatter_v11_chunk_ready_lists_kernel,
                dim3(ready_order_count),
                dim3(1),
                0,
                stream,
                rank_tile_problem_info,
                ntiles,
                static_cast<int *>(v11_chunk_tile_info),
                static_cast<int *>(v11_chunk_problem_info),
                static_cast<const int *>(v11_chunk_ready_offsets),
                world_size,
                rank,
                chunks_per_rank,
                v11_prefix_experts,
                v11_chunk_stage_major,
                max_tiles);
            err = hipGetLastError();
            if (err == hipSuccess)
            {
                err = hipMemcpyAsync(&problem_count,
                                     ntiles,
                                     sizeof(int),
                                     hipMemcpyDeviceToHost,
                                     stream);
            }
            if (err != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                return false;
            }
            cache.rank_tile_problem_info = rank_tile_problem_info;
            cache.ntiles = ntiles;
            cache.chunk_tile_info = v11_chunk_tile_info;
            cache.chunk_problem_info = v11_chunk_problem_info;
            cache.max_tiles = max_tiles;
            cache.world_size = world_size;
            cache.rank = rank;
            cache.chunks_per_rank = chunks_per_rank;
            cache.prefix_experts = v11_prefix_experts;
            cache.stage_major = v11_chunk_stage_major;
            cache.problem_count = problem_count;
            cache.valid = true;
        }
        original_problem_count = problem_count;
        active_rank_tile_info =
            static_cast<const int *>(v11_chunk_tile_info);
        active_rank_tile_problem_info =
            static_cast<const int *>(v11_chunk_problem_info);
    }
    else if (v11_enabled && !v11_chunk_signal)
    {
        const size_t max_problem_capacity = std::max(max_tiles, 1);
        const size_t stage_info_bytes =
            max_problem_capacity * AG_RANK_TILE_INFO_STRIDE * sizeof(int);
        if (!ag_mt256_ensure_scratch(&g_mt256_v11_stage_tile_info,
                                     stage_info_bytes,
                                     &v11_stage_tile_info) ||
            !ag_mt256_ensure_scratch(&g_mt256_v11_split_counts,
                                     2 * sizeof(int),
                                     &v11_split_counts) ||
            !ag_mt256_ensure_scratch(
                &g_mt256_v11_stage_cursors,
                2 * MAX_LOCAL_WORLD_SIZE * sizeof(int),
                &v11_stage_cursors))
        {
            return false;
        }
        const int window_count =
            (kMaxExpertsForContract + v11_window_experts - 1) /
            v11_window_experts;
        const int window_bucket_count = window_count * world_size;
        if (v11_window_stage_major &&
            !ag_mt256_ensure_scratch(
                &g_mt256_v11_window_offsets,
                static_cast<size_t>(window_bucket_count) * sizeof(int),
                &v11_window_offsets))
        {
            return false;
        }

        AgMt256V11StageCache &cache = g_mt256_v11_stage_cache;
        const bool stage_cache_hit =
            metadata_cache_hit && cache.valid &&
            cache.rank_tile_info == rank_tile_info && cache.ntiles == ntiles &&
            cache.stage_tile_info == v11_stage_tile_info &&
            cache.max_tiles == max_tiles && cache.world_size == world_size &&
            cache.rank == rank && cache.early_stages == v11_early_stages &&
            cache.prefix_experts == v11_prefix_experts &&
            cache.window_experts == v11_window_experts &&
            cache.window_stage_major == v11_window_stage_major &&
            cache.stage_major == v11_stage_major &&
            cache.split_launch == v11_split_launch;
        if (stage_cache_hit)
        {
            problem_count = cache.problem_count;
            v11_first_problem_count = cache.first_problem_count;
        }
        else
        {
            cache.valid = false;
            err = hipMemsetAsync(
                v11_stage_cursors,
                0,
                2 * MAX_LOCAL_WORLD_SIZE * sizeof(int),
                stream);
            if (err != hipSuccess)
            {
                return false;
            }
            if (v11_window_stage_major)
            {
                err = hipMemsetAsync(
                    v11_window_offsets,
                    0,
                    static_cast<size_t>(window_bucket_count) * sizeof(int),
                    stream);
                if (err != hipSuccess)
                {
                    return false;
                }
            }
            err = hipMemsetAsync(
                v11_split_counts, 0, 2 * sizeof(int), stream);
            if (err != hipSuccess)
            {
                return false;
            }
            constexpr int reorder_threads = 256;
            const int reorder_blocks =
                std::max((max_tiles + reorder_threads - 1) /
                             reorder_threads,
                         1);
            if (v11_window_stage_major)
            {
                hipLaunchKernelGGL(
                    ag_mt256_count_v11_window_stage_lists_kernel,
                    dim3(reorder_blocks),
                    dim3(reorder_threads),
                    0,
                    stream,
                    rank_tile_info,
                    ntiles,
                    static_cast<int *>(v11_window_offsets),
                    v11_window_experts,
                    world_size,
                    rank,
                    max_tiles);
                hipLaunchKernelGGL(
                    ag_mt256_prefix_v11_window_stage_lists_kernel,
                    dim3(1),
                    dim3(1),
                    0,
                    stream,
                    static_cast<int *>(v11_window_offsets),
                    window_bucket_count,
                    static_cast<int *>(v11_split_counts));
                hipLaunchKernelGGL(
                    ag_mt256_scatter_v11_window_stage_lists_kernel,
                    dim3(window_bucket_count),
                    dim3(1),
                    0,
                    stream,
                    rank_tile_info,
                    ntiles,
                    static_cast<int *>(v11_stage_tile_info),
                    static_cast<const int *>(v11_window_offsets),
                    v11_window_experts,
                    world_size,
                    rank,
                    max_tiles);
            }
            else if (v11_stage_major)
            {
                hipLaunchKernelGGL(
                    ag_mt256_count_v6_hybrid_tile_lists_kernel,
                    dim3(reorder_blocks),
                    dim3(reorder_threads),
                    0,
                    stream,
                    rank_tile_info,
                    ntiles,
                    static_cast<int *>(v11_stage_cursors),
                    0,
                    world_size,
                    rank,
                    true,
                    max_tiles);
                hipLaunchKernelGGL(
                    ag_mt256_prefix_v6_hybrid_tile_lists_kernel,
                    dim3(1),
                    dim3(1),
                    0,
                    stream,
                    static_cast<int *>(v11_stage_cursors),
                    static_cast<int *>(v11_split_counts),
                    world_size);
                hipLaunchKernelGGL(
                    ag_mt256_scatter_v6_hybrid_tile_lists_kernel,
                    dim3(2),
                    dim3(64),
                    0,
                    stream,
                    rank_tile_info,
                    ntiles,
                    static_cast<int *>(v11_stage_tile_info),
                    static_cast<int *>(v11_stage_tile_info),
                    static_cast<int *>(v11_stage_cursors),
                    0,
                    world_size,
                    rank,
                    true,
                    max_tiles);
            }
            else
            {
                hipLaunchKernelGGL(
                    ag_mt256_count_v11_early_late_lists_kernel,
                    dim3(reorder_blocks),
                    dim3(reorder_threads),
                    0,
                    stream,
                    rank_tile_info,
                    ntiles,
                    static_cast<int *>(v11_split_counts),
                    v11_early_stages,
                    v11_split_launch ? 256 : v11_prefix_experts,
                    world_size,
                    rank,
                    max_tiles);
                hipLaunchKernelGGL(
                    ag_mt256_scatter_v11_early_late_lists_kernel,
                    dim3(2),
                    dim3(1),
                    0,
                    stream,
                    rank_tile_info,
                    ntiles,
                    static_cast<int *>(v11_stage_tile_info),
                    static_cast<const int *>(v11_split_counts),
                    v11_early_stages,
                    v11_split_launch ? 256 : v11_prefix_experts,
                    world_size,
                    rank,
                    max_tiles);
            }
            err = hipGetLastError();
            int split_counts[2] = {0, 0};
            if (err == hipSuccess)
            {
                err = hipMemcpyAsync(split_counts,
                                     v11_split_counts,
                                     sizeof(split_counts),
                                     hipMemcpyDeviceToHost,
                                     stream);
            }
            if (err != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                return false;
            }
            problem_count = split_counts[0];
            if (!v11_stage_major)
            {
                v11_first_problem_count = split_counts[0];
                problem_count += split_counts[1];
            }
            cache.rank_tile_info = rank_tile_info;
            cache.ntiles = ntiles;
            cache.stage_tile_info = v11_stage_tile_info;
            cache.max_tiles = max_tiles;
            cache.world_size = world_size;
            cache.rank = rank;
            cache.early_stages = v11_early_stages;
            cache.prefix_experts = v11_prefix_experts;
            cache.window_experts = v11_window_experts;
            cache.window_stage_major = v11_window_stage_major;
            cache.stage_major = v11_stage_major;
            cache.split_launch = v11_split_launch;
            cache.problem_count = problem_count;
            cache.first_problem_count = v11_first_problem_count;
            cache.valid = true;
        }
        original_problem_count = problem_count;
        active_rank_tile_info =
            static_cast<const int *>(v11_stage_tile_info);
    }
    else if (v6_hybrid_tail)
    {
        const size_t max_problem_capacity =
            std::max(max_tiles, 1);
        const size_t main_info_bytes =
            max_problem_capacity * AG_RANK_TILE_INFO_STRIDE * sizeof(int);
        const size_t tail_info_bytes =
            max_problem_capacity * 4 * AG_RANK_TILE_INFO_STRIDE * sizeof(int);
        if (!ag_mt256_ensure_scratch(&g_mt256_v6_main_tile_info,
                                     main_info_bytes,
                                     &v6_main_tile_info) ||
            !ag_mt256_ensure_scratch(&g_mt256_v6_tail_tile_info,
                                     tail_info_bytes,
                                     &v6_tail_tile_info) ||
            !ag_mt256_ensure_scratch(&g_mt256_v6_split_counts,
                                     2 * sizeof(int),
                                     &v6_split_counts) ||
            !ag_mt256_ensure_scratch(
                &g_mt256_v6_stage_cursors,
                2 * MAX_LOCAL_WORLD_SIZE * sizeof(int),
                &v6_stage_cursors))
        {
            return false;
        }
        int max_tail_rows = get_env_int_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V6_MAX_TAIL_ROWS", 64);
        max_tail_rows = std::max(1, std::min(max_tail_rows, 255));
        AgMt256V6HybridCache &cache = g_mt256_v6_hybrid_cache;
        const bool split_cache_hit =
            metadata_cache_hit && cache.valid &&
            cache.rank_tile_info == rank_tile_info && cache.ntiles == ntiles &&
            cache.main_tile_info == v6_main_tile_info &&
            cache.tail_tile_info == v6_tail_tile_info &&
            cache.max_tail_rows == max_tail_rows &&
            cache.max_tiles == max_tiles && cache.world_size == world_size &&
            cache.rank == rank && cache.stage_major == v6_stage_major;
        if (split_cache_hit)
        {
            original_problem_count = cache.original_problem_count;
            problem_count = cache.main_problem_count;
            tail_problem_count = cache.tail_problem_count;
        }
        else
        {
            cache.valid = false;
            err = hipMemsetAsync(v6_stage_cursors,
                                 0,
                                 2 * MAX_LOCAL_WORLD_SIZE * sizeof(int),
                                 stream);
            if (err != hipSuccess)
            {
                return false;
            }
            constexpr int split_threads = 256;
            const int split_blocks =
                std::max((max_tiles + split_threads - 1) / split_threads, 1);
            hipLaunchKernelGGL(ag_mt256_count_v6_hybrid_tile_lists_kernel,
                               dim3(split_blocks),
                               dim3(split_threads),
                               0,
                               stream,
                               rank_tile_info,
                               ntiles,
                               static_cast<int *>(v6_stage_cursors),
                               max_tail_rows,
                               world_size,
                               rank,
                               v6_stage_major,
                               max_tiles);
            hipLaunchKernelGGL(ag_mt256_prefix_v6_hybrid_tile_lists_kernel,
                               dim3(1),
                               dim3(1),
                               0,
                               stream,
                               static_cast<int *>(v6_stage_cursors),
                               static_cast<int *>(v6_split_counts),
                               world_size);
            hipLaunchKernelGGL(ag_mt256_scatter_v6_hybrid_tile_lists_kernel,
                               dim3(2),
                               dim3(64),
                               0,
                               stream,
                               rank_tile_info,
                               ntiles,
                               static_cast<int *>(v6_main_tile_info),
                               static_cast<int *>(v6_tail_tile_info),
                               static_cast<int *>(v6_stage_cursors),
                               max_tail_rows,
                               world_size,
                               rank,
                               v6_stage_major,
                               max_tiles);
            err = hipGetLastError();
            if (err != hipSuccess)
            {
                return false;
            }
            int split_counts[2] = {0, 0};
            err = hipMemcpyAsync(&original_problem_count,
                                 ntiles,
                                 sizeof(int),
                                 hipMemcpyDeviceToHost,
                                 stream);
            if (err == hipSuccess)
            {
                err = hipMemcpyAsync(split_counts,
                                     v6_split_counts,
                                     sizeof(split_counts),
                                     hipMemcpyDeviceToHost,
                                     stream);
            }
            if (err != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                return false;
            }
            problem_count = split_counts[0];
            tail_problem_count = split_counts[1];
            cache.rank_tile_info = rank_tile_info;
            cache.ntiles = ntiles;
            cache.main_tile_info = v6_main_tile_info;
            cache.tail_tile_info = v6_tail_tile_info;
            cache.max_tail_rows = max_tail_rows;
            cache.max_tiles = max_tiles;
            cache.world_size = world_size;
            cache.rank = rank;
            cache.stage_major = v6_stage_major;
            cache.original_problem_count = original_problem_count;
            cache.main_problem_count = problem_count;
            cache.tail_problem_count = tail_problem_count;
            cache.valid = true;
        }
        active_rank_tile_info = static_cast<const int *>(v6_main_tile_info);
    }
    else
    {
        AgMt256V6ProblemCountCache &cache =
            g_mt256_v6_problem_count_cache;
        const bool cache_problem_count =
            v6_signal_gate || v11_chunk_signal || v12_ready_queue;
        const bool problem_count_cache_hit =
            cache_problem_count && metadata_cache_hit && cache.valid &&
            cache.rank_tile_info == rank_tile_info && cache.ntiles == ntiles &&
            cache.max_tiles == max_tiles;
        if (problem_count_cache_hit)
        {
            problem_count = cache.problem_count;
        }
        else
        {
            if (cache_problem_count)
            {
                cache.valid = false;
            }
            err = hipMemcpyAsync(&problem_count,
                                 ntiles,
                                 sizeof(int),
                                 hipMemcpyDeviceToHost,
                                 stream);
            if (err != hipSuccess ||
                hipStreamSynchronize(stream) != hipSuccess)
            {
                return false;
            }
            if (cache_problem_count)
            {
                cache.rank_tile_info = rank_tile_info;
                cache.ntiles = ntiles;
                cache.max_tiles = max_tiles;
                cache.problem_count = problem_count;
                cache.valid = true;
            }
        }
        original_problem_count = problem_count;
    }
    if (original_problem_count <= 0 || problem_count <= 0)
    {
        return false;
    }
    profile.record(1, stream);

    // V36 packed output defaults to the vector scatter path.  The store path
    // still falls back to scalar scatter when n_dim is not vector-aligned.
    const bool vec_scatter = !direct_output && get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_VEC_SCATTER", v36_dynamic);
    const bool vec_pack_requested = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_VEC_PACK", false);

    const int64_t packed_rows = static_cast<int64_t>(problem_count) * 256;
    const size_t packed_input_bytes =
        direct_input
            ? 0
            : static_cast<size_t>(packed_rows) * static_cast<size_t>(k_dim) *
                  sizeof(uint16_t);
    const size_t packed_output_bytes =
        direct_output
            ? 0
            : static_cast<size_t>(packed_rows) * static_cast<size_t>(n_dim) *
                  sizeof(uint16_t);
    const bool chunk_ready_contract =
        v11_chunk_signal || v36_rank_chunk_signal;
    const size_t tile_info_bytes = static_cast<size_t>(problem_count) *
                                   (chunk_ready_contract ? 7 : 5) * sizeof(int);
    const size_t offsets_bytes =
        static_cast<size_t>(kMaxExpertsForContract + 1) * sizeof(int);

    void *packed_input = nullptr;
    void *packed_output = nullptr;
    void *contract_tile_info = nullptr;
    void *sorted_expert_offsets = nullptr;
    void *v12_task_claims = nullptr;
    void *v36_direct_control = nullptr;
    int v36_direct_chunk_n = get_env_int_with_legacy(
        "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_DIRECT_CHUNK_N",
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_DIRECT_CHUNK_N",
        1024);
    v36_direct_chunk_n = std::max(256, std::min(v36_direct_chunk_n, n_dim));
    v36_direct_chunk_n = (v36_direct_chunk_n / 256) * 256;
    const int v36_direct_done_groups = v36_direct_output
        ? (n_dim + v36_direct_chunk_n - 1) / v36_direct_chunk_n
        : 0;
    const bool v36_direct_nchunk_signal = v36_direct_output &&
        n_chunk_done_counts != nullptr && n_chunk_done_flags != nullptr;
    const bool v36_direct_group_major = v36_direct_nchunk_signal &&
        !v36_expert_n_major_direct_output &&
        get_env_bool_with_legacy(
            "FTRAIN_HYGON_AG_GROUP_GEMM_RANK_TILE_DIRECT_GROUP_MAJOR",
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_V36_DIRECT_GROUP_MAJOR",
            false);
    const size_t v12_task_claim_bytes =
        static_cast<size_t>(problem_count) * static_cast<size_t>(n_tiles) *
        sizeof(int32_t);
    const size_t v36_direct_control_bytes = v36_direct_output
        ? (static_cast<size_t>(problem_count) *
               static_cast<size_t>(n_tiles) +
           static_cast<size_t>(v36_direct_done_groups)) *
              sizeof(int32_t)
        : 0;
    if ((!direct_input &&
         !ag_mt256_ensure_scratch(&g_mt256_native_packed_input,
                                  packed_input_bytes,
                                  &packed_input)) ||
        (!direct_output &&
         !ag_mt256_ensure_scratch(&g_mt256_native_packed_output,
                                  packed_output_bytes,
                                  &packed_output)) ||
        !ag_mt256_ensure_scratch(&g_mt256_native_tile_info,
                                 tile_info_bytes,
                                 &contract_tile_info) ||
        (v12_ready_queue && !v36_direct_output &&
         !ag_mt256_ensure_scratch(&g_mt256_v12_task_claims,
                                  v12_task_claim_bytes,
                                  &v12_task_claims)) ||
        (v36_direct_output &&
         !ag_mt256_ensure_scratch(&g_mt256_v36_direct_control,
                                  v36_direct_control_bytes,
                                  &v36_direct_control)) ||
        (!direct_input &&
         !ag_mt256_ensure_scratch(&g_mt256_native_expert_offsets,
                                  offsets_bytes,
                                  &sorted_expert_offsets)))
    {
        return false;
    }
    if (v12_ready_queue && !v36_expert_n_major && !v36_direct_output)
    {
        err = hipMemsetAsync(v12_task_claims,
                             0,
                             v12_task_claim_bytes,
                             stream);
        if (err != hipSuccess)
        {
            return false;
        }
    }
    if (v36_direct_output)
    {
        err = hipMemsetAsync(v36_direct_control,
                             0,
                             v36_direct_control_bytes,
                             stream);
        if (err != hipSuccess)
        {
            return false;
        }
    }

    constexpr int threads = 256;
    const int info_blocks = std::max(
        (std::max(problem_count, kMaxExpertsForContract + 1) + threads - 1) /
            threads,
        1);
    if (direct_input)
    {
        if (v36_rank_chunk_signal)
        {
            hipLaunchKernelGGL(
                ag_mt256_build_native_direct_input_rank_chunk_info_kernel,
                dim3(info_blocks),
                dim3(threads),
                0,
                stream,
                active_rank_tile_problem_info,
                static_cast<int *>(contract_tile_info),
                problem_count,
                chunks_per_rank,
                rank,
                world_size,
                get_env_int_local(
                    "FTRAIN_HYGON_AG_GROUP_GEMM_RAGGED_CHUNK_SLOT_BASE", 0));
        }
        else if (chunk_ready_contract)
        {
            hipLaunchKernelGGL(
                ag_mt256_build_native_direct_input_chunk_info_kernel,
                dim3(info_blocks),
                dim3(threads),
                0,
                stream,
                active_rank_tile_problem_info,
                static_cast<int *>(contract_tile_info),
                problem_count);
        }
        else if (v11_split_launch && v11_first_problem_count > 0 &&
            v11_first_problem_count < problem_count)
        {
            hipLaunchKernelGGL(
                ag_mt256_build_native_direct_input_split_info_kernel,
                dim3(info_blocks),
                dim3(threads),
                0,
                stream,
                active_rank_tile_info,
                static_cast<int *>(contract_tile_info),
                problem_count,
                v11_first_problem_count);
        }
        else
        {
            printf("[hygon] native contract info rank=%d problems=%d ragged=%d "
                   "stage_layout=%d v36_stage_ordered=%d\n",
                   rank,
                   problem_count,
                   ragged_m ? 1 : 0,
                   (v36_stage_ordered ||
                    get_env_bool_local(
                        "FTRAIN_HYGON_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT",
                        false))
                       ? 1
                       : 0,
                   v36_stage_ordered ? 1 : 0);
            hipLaunchKernelGGL(ag_mt256_build_native_direct_input_info_kernel,
                               dim3(info_blocks),
                               dim3(threads),
                               0,
                               stream,
                               active_rank_tile_info,
                               static_cast<int *>(contract_tile_info),
                               problem_count,
                               sorted_gather_index,
                               full_m,
                               world_size,
                               compact_physical_rows,
                               rank,
                               v36_stage_ordered ||
                                   get_env_bool_local(
                                       "FTRAIN_HYGON_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT",
                                       false),
                               ragged_m);
        }
    }
    else
    {
        hipLaunchKernelGGL(ag_mt256_build_native_contract_info_kernel,
                           dim3(info_blocks),
                           dim3(threads),
                           0,
                           stream,
                           active_rank_tile_info,
                           static_cast<int *>(contract_tile_info),
                           static_cast<int *>(sorted_expert_offsets),
                           problem_count,
                           kMaxExpertsForContract);
    }
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        return false;
    }
    if (v36_rank_chunk_signal && rank == 0 &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_DEBUG_RANK_CHUNK_INFO", false))
    {
        std::vector<int> host_contract(static_cast<size_t>(problem_count) * 5,
                                       0);
        CUDA_CHECK(hipMemcpyAsync(host_contract.data(),
                                  contract_tile_info,
                                  host_contract.size() * sizeof(int),
                                  hipMemcpyDeviceToHost,
                                  stream));
        CUDA_CHECK(hipStreamSynchronize(stream));
        const int count = std::min(problem_count, 16);
        for (int index = 0; index < count; ++index)
        {
            printf("[hygon] rank-chunk contract tile=%d expert=%d start=%d "
                   "chunk=%d..%d rows=%d\n",
                   index,
                   host_contract[index],
                   host_contract[problem_count + index],
                   host_contract[2 * problem_count + index],
                   host_contract[3 * problem_count + index],
                   host_contract[4 * problem_count + index]);
        }
        fflush(stdout);
    }
    if (v36_expert_n_major)
    {
        hipLaunchKernelGGL(ag_mt256_build_expert_n_major_task_map_kernel,
                           dim3(info_blocks),
                           dim3(threads),
                           0,
                           stream,
                           active_rank_tile_info,
                           static_cast<int32_t *>(
                               v36_expert_n_major_direct_output
                                   ? v36_direct_control
                                   : v12_task_claims),
                           problem_count,
                           n_tiles,
                           v36_expert_n_major_readiness_window);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }
    }
    profile.record(2, stream);

    hipFunction_t function = nullptr;
    const bool force_v5_wgm8_fallback =
        v36_stage_ordered_requested && !v36_stage_ordered;
    if (!ag_mt256_get_native_contract_function(
            &function,
            force_v5_wgm8_fallback,
            v36_dynamic,
            v36_fixed_n8,
            v36_expert_n_major,
            v36_expert_n_major_direct_output,
            ragged_m))
    {
        FTRAIN_HYGON_CHECK(!v36_stage_ordered)
            << "V36 stage-layout metadata requires its native ASM kernel";
        return false;
    }

    AgMt256ContractRankTileArgs args{};
    args.input_buffer = direct_input
                            ? static_cast<const uint16_t *>(full_input)
                            : static_cast<const uint16_t *>(packed_input);
    args.weight = static_cast<const uint16_t *>(weight);
    args.sorted_expert_offsets = direct_input
        ? static_cast<const int32_t *>(sorted_gather_index)
        : static_cast<const int32_t *>(sorted_expert_offsets);
    args.output = static_cast<uint16_t *>(
        direct_output ? output : packed_output);
    args.barrier_ptr = rank_signals;
    args.tile_info = static_cast<const int32_t *>(contract_tile_info);
    args.problem_count = static_cast<uint32_t>(problem_count);
    args.n_dim = static_cast<uint32_t>(n_dim);
    args.k_dim = static_cast<uint32_t>(k_dim);
    args.topk = static_cast<uint32_t>(topk);
    args.rank = (v36_stage_ordered ||
                 get_env_bool_local(
                     "FTRAIN_HYGON_AG_GROUP_GEMM_PUSH_STAGE_SIGNAL_LAYOUT", false))
        ? 0u
        : static_cast<uint32_t>(rank);
    args.wait_signal_value = static_cast<uint32_t>(
        v6_tile_wait ? signal_value : (v6_signal_gate ? 0 : signal_value));
    args.epilogue_mode = direct_output ? 1u : 0u;
    args.chunk_size_n = v36_direct_nchunk_signal
        ? static_cast<uint32_t>(v36_direct_chunk_n)
        : (chunk_ready_contract ? 1u : 0u);
    args.num_done_groups = v36_direct_nchunk_signal
        ? static_cast<uint32_t>(v36_direct_done_groups)
        : (chunk_ready_contract ? static_cast<uint32_t>(chunks_per_rank) : 0u);
    // reserved bit 0 is the v36 group-major launch mode.  Bit 5 selects the
    // compact physical-row encoding in the NN/Ailk direct-input body.
    args.reserved = (v36_direct_group_major ? 1u : 0u) |
                    (compact_physical_rows ? 32u : 0u) |
                    ((ragged_ring_chunk || v36_rank_chunk_signal) ? 128u : 0u);
    args.topk_stage = direct_output
                          ? const_cast<int *>(sorted_token_ids)
                          : nullptr;
    args.chunk_done_counters = v36_direct_output
        ? static_cast<int32_t *>(v36_direct_control)
        : (v12_ready_queue ? static_cast<int32_t *>(v12_task_claims)
                           : nullptr);
    args.chunk_done_flags = v36_direct_nchunk_signal
        ? n_chunk_done_flags
        : nullptr;
    profile.record(3, stream);

    bool vec_pack_active = false;
    if (direct_input)
    {
        vec_pack_active = false;
    }
    else if (tile_wait_pack)
    {
        const int k_tile = get_env_int_local(
            "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TILE_WAIT_PACK_K", 512);
        const int active_k_tile = k_tile > 0 ? k_tile : 512;
        const bool use_vec_pack =
            vec_pack_requested && (k_dim % 8 == 0) &&
            (active_k_tile % 8 == 0);
        vec_pack_active = use_vec_pack;
        if (use_vec_pack)
        {
            const int k_vec_dim = k_dim / 8;
            const int k_tile_vec = active_k_tile / 8;
            const int k_slices =
                (k_vec_dim + k_tile_vec - 1) / k_tile_vec;
            hipLaunchKernelGGL(
                ag_mt256_pack_native_rank_tiles_tile_wait_vec8_kernel,
                dim3(problem_count, k_slices),
                dim3(threads),
                0,
                stream,
                static_cast<const uint4 *>(full_input),
                sorted_gather_index,
                active_rank_tile_info,
                rank_signals,
                static_cast<uint4 *>(packed_input),
                problem_count,
                full_m,
                k_vec_dim,
                world_size,
                signal_value,
                k_tile_vec);
        }
        else
        {
            const int k_slices = (k_dim + active_k_tile - 1) / active_k_tile;
            hipLaunchKernelGGL(ag_mt256_pack_native_rank_tiles_tile_wait_kernel,
                               dim3(problem_count, k_slices),
                               dim3(threads),
                               0,
                               stream,
                               static_cast<const uint16_t *>(full_input),
                               sorted_gather_index,
                               active_rank_tile_info,
                               rank_signals,
                               static_cast<uint16_t *>(packed_input),
                               problem_count,
                               full_m,
                               k_dim,
                               world_size,
                               signal_value,
                               active_k_tile);
        }
    }
    else
    {
        const int64_t pack_total = packed_rows * k_dim;
        const int pack_blocks = std::max(
            1,
            static_cast<int>(std::min<int64_t>(
                (pack_total + threads - 1) / threads, 4096)));
        hipLaunchKernelGGL(ag_mt256_pack_native_rank_tiles_kernel,
                           dim3(pack_blocks),
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const uint16_t *>(full_input),
                           sorted_gather_index,
                           active_rank_tile_info,
                           static_cast<uint16_t *>(packed_input),
                           problem_count,
                           full_m,
                           k_dim);
    }
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        return false;
    }
    profile.record(4, stream);

    size_t arg_size = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &arg_size,
                      HIP_LAUNCH_PARAM_END};
    const bool tnloader_producer8 = get_env_bool_local(
        "FTRAIN_HYGON_AG_GROUP_GEMM_ASM_MT256_TNLOADER_PRODUCER8", false);
    const uint32_t native_block_threads = tnloader_producer8 ? 1024u : 768u;
    if (v6_signal_gate && !v6_tile_wait && signal_value > 0)
    {
        hipLaunchKernelGGL(wait_rank_signals_system_kernel,
                           dim3(1),
                           dim3(1),
                           0,
                           stream,
                           rank_signals,
                           world_size,
                           signal_value);
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }
    }
    const auto launch_native_contract = [&](uint32_t active_problems) {
        args.problem_count = active_problems;
        const uint32_t v36_direct_tiles_per_group =
            static_cast<uint32_t>(v36_direct_chunk_n / 256);
        const uint32_t launch_groups_x =
            v36_direct_group_major
                ? v36_direct_tiles_per_group
            : v36_flat_grid
                ? static_cast<uint32_t>(n_tiles) * active_problems
                : (wgm8 ? static_cast<uint32_t>(n_tiles) : active_problems);
        const uint32_t launch_groups_y =
            v36_direct_group_major
                ? active_problems *
                      static_cast<uint32_t>(v36_direct_done_groups)
            : v36_flat_grid
                ? 1u
                : (wgm8 ? active_problems : static_cast<uint32_t>(n_tiles));
        return hipExtModuleLaunchKernel(function,
                                        launch_groups_x * native_block_threads,
                                        launch_groups_y,
                                        1,
                                        native_block_threads,
                                        1,
                                        1,
                                        0,
                                        stream,
                                        nullptr,
                                        config,
                                        0,
                                        0,
                                        0);
    };
    if (v11_split_launch && v11_first_problem_count > 0 &&
             v11_first_problem_count < problem_count)
    {
        const int second_problem_count =
            problem_count - v11_first_problem_count;
        args.problem_count = static_cast<uint32_t>(v11_first_problem_count);
        err = launch_native_contract(args.problem_count);
        if (err == hipSuccess && signal_value > 0)
        {
            hipLaunchKernelGGL(wait_rank_signals_system_kernel,
                               dim3(1),
                               dim3(1),
                               0,
                               stream,
                               rank_signals,
                               world_size,
                               signal_value);
            err = hipGetLastError();
        }
        if (err == hipSuccess)
        {
            args.output = static_cast<uint16_t *>(packed_output) +
                          static_cast<int64_t>(v11_first_problem_count) *
                              256 * n_dim;
            args.tile_info = static_cast<const int32_t *>(contract_tile_info) +
                             5 * v11_first_problem_count;
            args.wait_signal_value = 0;
            err = launch_native_contract(
                static_cast<uint32_t>(second_problem_count));
        }
    }
    else
    {
        err = launch_native_contract(static_cast<uint32_t>(problem_count));
    }
    if (err != hipSuccess)
    {
        return false;
    }
    if (v6_hybrid_tail && tail_problem_count > 0)
    {
        if (v6_tail_vec8)
        {
            hipLaunchKernelGGL(
                (ag_group_gemm_tail_64x256x32_lds_mmac_bf16_nk_load_vec_kernel<8>),
                dim3(n_tiles, tail_problem_count),
                dim3(256),
                0,
                stream,
                static_cast<const uint16_t *>(full_input),
                static_cast<const uint16_t *>(weight),
                sorted_token_ids,
                sorted_gather_index,
                static_cast<const int *>(v6_tail_tile_info),
                static_cast<const int *>(v6_split_counts) + 1,
                rank_signals,
                static_cast<uint16_t *>(output),
                full_m,
                n_dim,
                k_dim,
                topk,
                world_size,
                v6_tile_wait ? signal_value : 0);
        }
        else
        {
            hipLaunchKernelGGL(
                (ag_group_gemm_tail_64x256x32_lds_mmac_bf16_nk_load_vec_kernel<4>),
                dim3(n_tiles, tail_problem_count),
                dim3(256),
                0,
                stream,
                static_cast<const uint16_t *>(full_input),
                static_cast<const uint16_t *>(weight),
                sorted_token_ids,
                sorted_gather_index,
                static_cast<const int *>(v6_tail_tile_info),
                static_cast<const int *>(v6_split_counts) + 1,
                rank_signals,
                static_cast<uint16_t *>(output),
                full_m,
                n_dim,
                k_dim,
                topk,
                world_size,
                v6_tile_wait ? signal_value : 0);
        }
        err = hipGetLastError();
        if (err != hipSuccess)
        {
            return false;
        }
    }
    profile.record(5, stream);

    if (direct_output)
    {
        // The assembly epilogue has already row-mapped every value into output.
    }
    else if (vec_scatter && n_dim % 8 == 0)
    {
        const int n_vec = n_dim / 8;
        const int64_t scatter_total = packed_rows * n_vec;
        const int scatter_blocks = std::max(
            1,
            static_cast<int>(std::min<int64_t>(
                (scatter_total + threads - 1) / threads, 4096)));
        hipLaunchKernelGGL(ag_mt256_scatter_native_contract_output_vec8_kernel,
                           dim3(scatter_blocks),
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const uint4 *>(packed_output),
                           sorted_token_ids,
                           active_rank_tile_info,
                           static_cast<uint4 *>(output),
                           problem_count,
                           full_m * topk,
                           n_vec);
    }
    else
    {
        const int64_t scatter_total = packed_rows * n_dim;
        const int scatter_blocks = std::max(
            1,
            static_cast<int>(std::min<int64_t>(
                (scatter_total + threads - 1) / threads, 4096)));
        hipLaunchKernelGGL(ag_mt256_scatter_native_contract_output_kernel,
                           dim3(scatter_blocks),
                           dim3(threads),
                           0,
                           stream,
                           static_cast<const uint16_t *>(packed_output),
                           sorted_token_ids,
                           active_rank_tile_info,
                           static_cast<uint16_t *>(output),
                           problem_count,
                           full_m * topk,
                           n_dim);
    }
    err = hipGetLastError();
    if (err != hipSuccess)
    {
        return false;
    }
    profile.record(6, stream);
    profile.print(rank, profile_call_index);

    static std::atomic<bool> logged{false};
    bool expected = false;
    if (rank == 0 && logged.compare_exchange_strong(expected, true))
    {
        printf("[hygon] ASM MT256 native contract enabled: problems=%d "
               "packed_rows=%lld n=%d k=%d direct_input=%d "
               "tile_wait_pack=%d vec_pack=%d vec_scatter=%d "
               "direct_output=%d lds_rowmap=%d oob_store=%d wide_n=%d "
               "max_n_tiles=%d v6_signal_gate=%d v6_hybrid_tail=%d "
               "v6_tile_wait=%d v6_stage_major=%d v6_tail_vec8=%d "
               "v11_stage_major=%d v11_split_launch=%d "
               "v11_ready_prefix=%d v11_early_stages=%d "
               "v11_prefix_experts=%d v11_first_problems=%d "
               "v11_window_stage_major=%d v11_window_experts=%d "
               "v11_chunk_signal=%d v11_chunk_stage_major=%d "
               "v11_chunk_ready_prefix=%d "
               "v12_ready_queue=%d v36_dynamic=%d v36_fixed_n8=%d "
               "v36_flat_grid=%d v36_expert_n_major=%d "
               "v36_expert_direct64=%d "
               "v36_expert_stage_major=%d v36_expert_ready_window=%d "
               "chunk_rows=%d chunks_per_rank=%d "
               "original_problems=%d "
               "tail_problems=%d\n",
               problem_count,
               static_cast<long long>(packed_rows),
               n_dim,
               k_dim,
               direct_input ? 1 : 0,
               tile_wait_pack ? 1 : 0,
               vec_pack_active ? 1 : 0,
               (vec_scatter && n_dim % 8 == 0) ? 1 : 0,
               direct_output ? 1 : 0,
               lds_rowmap ? 1 : 0,
               oob_store ? 1 : 0,
               allow_wide_n ? 1 : 0,
               max_wgm8_n_tiles,
               v6_signal_gate ? 1 : 0,
               v6_hybrid_tail ? 1 : 0,
               v6_tile_wait ? 1 : 0,
               v6_stage_major ? 1 : 0,
               v6_tail_vec8 ? 1 : 0,
               v11_stage_major ? 1 : 0,
               v11_split_launch ? 1 : 0,
               v11_ready_prefix ? 1 : 0,
               v11_early_stages,
               v11_prefix_experts,
               v11_first_problem_count,
               v11_window_stage_major ? 1 : 0,
               v11_window_experts,
               v11_chunk_signal ? 1 : 0,
               v11_chunk_stage_major ? 1 : 0,
               v11_chunk_ready_prefix ? 1 : 0,
               v12_ready_queue ? 1 : 0,
               v36_dynamic ? 1 : 0,
               v36_fixed_n8 ? 1 : 0,
               v36_flat_grid ? 1 : 0,
               v36_expert_n_major ? 1 : 0,
               v36_expert_n_major_direct_output ? 1 : 0,
               v36_expert_n_major_stage_major ? 1 : 0,
               v36_expert_n_major_readiness_window,
               signal_chunk_rows,
               chunks_per_rank,
               original_problem_count,
               tail_problem_count);
    }
    return true;
}

void ag_group_gemm_tiled_forward_impl(const void *full_input,
                                      const void *weight,
                                      const int *sorted_token_ids,
                                      const int *sorted_gather_index,
                                      const int *rank_tile_info,
                                      const int *tile_experts,
                                      const int *tile_starts,
                                      const int *tile_rank_starts,
                                      const int *tile_rank_ends,
                                      const int *rank_tile_problem_info,
                                      const int *ntiles,
                                      const int *rank_signals,
                                      int *rank_tile_ready_flags,
                                      int *n_chunk_done_counts,
                                      int *n_chunk_done_flags,
                                      void *output,
                                      int full_m,
                                      int n_dim,
                                      int k_dim,
                                      int topk,
                                      int num_experts,
                                      int world_size,
                                      int rank,
                                      int max_tiles,
                                      bool weight_nk,
                                      bool metadata_cache_hit,
                                      bool ragged_m,
                                      int signal_value,
                                      int signal_chunk_rows,
                                      int chunks_per_rank,
                                      c10::ScalarType dtype,
                                      hipStream_t stream)
{
    if (ag_mt256_launch_native_contract(full_input,
                                        weight,
                                        sorted_token_ids,
                                        sorted_gather_index,
                                        rank_tile_info,
                                        rank_tile_problem_info,
                                        ntiles,
                                        rank_signals,
                                        n_chunk_done_counts,
                                        n_chunk_done_flags,
                                        output,
                                        full_m,
                                        n_dim,
                                        k_dim,
                                        topk,
                                        num_experts,
                                        world_size,
                                        rank,
                                        max_tiles,
                                        weight_nk,
                                        metadata_cache_hit,
                                        ragged_m,
                                        signal_value,
                                        signal_chunk_rows,
                                        chunks_per_rank,
                                        dtype,
                                        stream))
    {
        return;
    }

    constexpr int BN = 256;
    constexpr int SUB_BN = 128;
    constexpr int MMAC_BN = 128;
    const bool use_mmac_direct =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_DIRECT_MMAC", true);
    const bool use_lds_mmac =
        (dtype == c10::ScalarType::Half ||
         dtype == c10::ScalarType::BFloat16) &&
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_LDS_MMAC", true);
    const bool use_smem_tile =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_REBUILT_SMEM_TILE", true);
    const bool use_mmac_shmem_b =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_MMAC_SHMEM_B", true);
    const int direct_bn =
        (use_mmac_direct && use_lds_mmac && dtype == c10::ScalarType::BFloat16)
            ? 256
            : MMAC_BN;
    const int n_tiles = use_mmac_direct
                            ? ((n_dim + direct_bn - 1) / direct_bn)
                            : (use_smem_tile ? ((n_dim + SUB_BN - 1) / SUB_BN)
                                             : ((n_dim + BN - 1) / BN));
    const dim3 grid(use_mmac_direct ? max_tiles
                                    : (use_smem_tile ? max_tiles * 2
                                                     : max_tiles),
                    n_tiles);

    if (dtype == c10::ScalarType::BFloat16)
    {
        if (use_mmac_direct && use_lds_mmac)
        {
            const bool use_source_dsread_n128 = get_env_bool_local(
                "FTRAIN_HYGON_AG_GROUP_GEMM_SOURCE_DSREAD_N128",
                true);
            const bool use_dsread_fragment_layout = get_env_bool_local(
                "FTRAIN_HYGON_AG_GROUP_GEMM_DSREAD_FRAGMENT_LAYOUT",
                use_source_dsread_n128);
            const bool use_dsread_swizzle =
                use_dsread_fragment_layout &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_DSREAD_SWIZZLE",
                                   false);
            const bool use_wg8_fragment =
                use_dsread_fragment_layout &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG8_FRAGMENT", false);
            const bool use_wg8_m32 =
                !use_dsread_fragment_layout &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG8_M32", false);
            const bool prefer_wg16_m32_n64 =
                !use_source_dsread_n128 && !use_dsread_fragment_layout &&
                !use_wg8_m32 &&
                n_dim >= 512 && k_dim >= 512;
            const bool use_wg16_m32_n64 =
                !use_dsread_fragment_layout && !use_wg8_m32 &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG16_M32_N64",
                                   prefer_wg16_m32_n64);
            const bool use_wg16_pair_b =
                use_wg16_m32_n64 &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WG16_M32_N64_PAIRB",
                                   false);
            const bool prefer_tile_ready =
                use_source_dsread_n128 && world_size > 1 && n_dim > 256 &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_SOURCE_TILE_READY",
                                   false);
            const bool use_tile_ready =
                rank_signals != nullptr && rank_tile_ready_flags != nullptr &&
                get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_TILE_READY_FLAG",
                                   prefer_tile_ready);
            if (use_wg8_fragment)
            {
                if (use_tile_ready)
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                true,
                                true>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                true,
                                false>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                else
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                false,
                                true>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_fragment_kernel<
                                false,
                                false>),
                            grid,
                            dim3(512),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                return;
            }
            if (use_wg8_m32)
            {
                if (use_tile_ready)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_m32_kernel<
                            true>),
                        grid,
                        dim3(512),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg8_m32_kernel<
                            false>),
                        grid,
                        dim3(512),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
            else if (use_wg16_m32_n64)
            {
                if (use_tile_ready && use_wg16_pair_b)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            true,
                            true>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else if (use_tile_ready)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            true,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else if (use_wg16_pair_b)
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            false,
                            true>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_wg16_m32_n64_kernel<
                            false,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
            else if (use_tile_ready)
            {
                if (use_dsread_fragment_layout)
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                true,
                                true,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                true,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            rank_tile_ready_flags,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                            true,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        rank_tile_ready_flags,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
            else
            {
                if (use_dsread_fragment_layout)
                {
                    if (use_dsread_swizzle)
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                false,
                                true,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                    else
                    {
                        hipLaunchKernelGGL(
                            (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                                false,
                                true>),
                            grid,
                            dim3(1024),
                            0,
                            stream,
                            static_cast<const uint16_t *>(full_input),
                            static_cast<const uint16_t *>(weight),
                            sorted_token_ids,
                            sorted_gather_index,
                            rank_tile_info,
                            rank_tile_problem_info,
                            ntiles,
                            rank_signals,
                            nullptr,
                            n_chunk_done_counts,
                            n_chunk_done_flags,
                            static_cast<uint16_t *>(output),
                            full_m,
                            n_dim,
                            k_dim,
                            topk,
                            world_size,
                            signal_value,
                            signal_chunk_rows,
                            chunks_per_rank);
                    }
                }
                else
                {
                    hipLaunchKernelGGL(
                        (ag_group_gemm_tiled_128x256x32_lds_mmac_bf16_kernel<
                            false,
                            false>),
                        grid,
                        dim3(1024),
                        0,
                        stream,
                        static_cast<const uint16_t *>(full_input),
                        static_cast<const uint16_t *>(weight),
                        sorted_token_ids,
                        sorted_gather_index,
                        rank_tile_info,
                        rank_tile_problem_info,
                        ntiles,
                        rank_signals,
                        nullptr,
                        n_chunk_done_counts,
                        n_chunk_done_flags,
                        static_cast<uint16_t *>(output),
                        full_m,
                        n_dim,
                        k_dim,
                        topk,
                        world_size,
                        signal_value,
                        signal_chunk_rows,
                        chunks_per_rank);
                }
            }
        }
        else if (use_mmac_direct)
        {
            hipLaunchKernelGGL(ag_group_gemm_tiled_128x128x32_mmac_bf16_kernel,
                               grid,
                               dim3(512),
                               0,
                               stream,
                               static_cast<const uint16_t *>(full_input),
                               static_cast<const uint16_t *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<uint16_t *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               weight_nk,
                               signal_value);
        }
        else if (use_smem_tile)
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_smem64x128_kernel<
                                   __hip_bfloat16>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __hip_bfloat16 *>(full_input),
                               static_cast<const __hip_bfloat16 *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__hip_bfloat16 *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
        else
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_kernel<
                                   __hip_bfloat16>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __hip_bfloat16 *>(full_input),
                               static_cast<const __hip_bfloat16 *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__hip_bfloat16 *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
    }
    else if (dtype == c10::ScalarType::Half)
    {
        if (use_mmac_direct && use_lds_mmac)
        {
            hipLaunchKernelGGL(
                ag_group_gemm_tiled_128x128x16_lds_mmac_f16_kernel,
                grid,
                dim3(1024),
                0,
                stream,
                static_cast<const _Float16 *>(full_input),
                static_cast<const _Float16 *>(weight),
                sorted_token_ids,
                tile_experts,
                tile_starts,
                tile_rank_starts,
                tile_rank_ends,
                ntiles,
                rank_signals,
                static_cast<_Float16 *>(output),
                full_m,
                n_dim,
                k_dim,
                topk,
                world_size,
                signal_value);
        }
        else if (use_mmac_direct)
        {
            hipLaunchKernelGGL(ag_group_gemm_tiled_128x128x32_mmac_f16_kernel,
                               grid,
                               dim3(512),
                               0,
                               stream,
                               static_cast<const _Float16 *>(full_input),
                               static_cast<const _Float16 *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<_Float16 *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               weight_nk,
                               signal_value,
                               use_mmac_shmem_b);
        }
        else if (use_smem_tile)
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_smem64x128_kernel<
                                   __half>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __half *>(full_input),
                               static_cast<const __half *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__half *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
        else
        {
            hipLaunchKernelGGL((ag_group_gemm_tiled_128x256x32_kernel<__half>),
                               grid,
                               dim3(256),
                               0,
                               stream,
                               static_cast<const __half *>(full_input),
                               static_cast<const __half *>(weight),
                               sorted_token_ids,
                               tile_experts,
                               tile_starts,
                               tile_rank_starts,
                               tile_rank_ends,
                               ntiles,
                               rank_signals,
                               static_cast<__half *>(output),
                               full_m,
                               n_dim,
                               k_dim,
                               topk,
                               world_size,
                               signal_value);
        }
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "AGGroupGemm tiled path supports FP16/BF16";
    }
}

void ag_group_gemm_pack_weight_impl(const void *weight,
                                    void *packed_weight,
                                    int num_experts,
                                    int k_dim,
                                    int n_dim,
                                    c10::ScalarType dtype,
                                    hipStream_t stream)
{
    constexpr int tile_dim = 32;
    constexpr int block_rows = 8;
    const dim3 grid((n_dim + tile_dim - 1) / tile_dim,
                    (k_dim + tile_dim - 1) / tile_dim,
                    num_experts);
    const dim3 block(tile_dim, block_rows);
    if (dtype == c10::ScalarType::BFloat16)
    {
        hipLaunchKernelGGL((pack_weight_kn_to_nk_kernel<uint16_t>),
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const uint16_t *>(weight),
                           static_cast<uint16_t *>(packed_weight),
                           k_dim,
                           n_dim);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        hipLaunchKernelGGL((pack_weight_kn_to_nk_kernel<_Float16>),
                           grid,
                           block,
                           0,
                           stream,
                           static_cast<const _Float16 *>(weight),
                           static_cast<_Float16 *>(packed_weight),
                           k_dim,
                           n_dim);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "AGGroupGemm weight pack supports FP16/BF16";
    }
}

void ag_group_gemm_backward_weight_impl(
    const void *full_routed_input,
    const void *gathered_grad_output,
    const int *sorted_token_ids,
    const int *sorted_gather_index,
    const int *global_expert_offsets,
    const int *rank_cursors,
    void *compact_routed_input,
    void *compact_grad_output,
    void *grouped_asm_args,
    void *grad_weight,
    int sorted_capacity,
    int full_m,
    int output_dim,
    int reduction_dim,
    int num_experts,
    int topk,
    int world_size,
    int signal_value,
    const int *output_chunk_ready_flags,
    int output_chunk_size,
    c10::ScalarType dtype,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(dtype == c10::ScalarType::BFloat16)
        << "AGGroupGemm backward_weight currently supports BF16";
    FTRAIN_HYGON_CHECK(compact_routed_input != nullptr && compact_grad_output != nullptr)
        << "AGGroupGemm backward_weight requires compact operand workspaces";
    FTRAIN_HYGON_CHECK(grouped_asm_args != nullptr)
        << "AGGroupGemm backward_weight requires grouped ASM args workspace";

    const bool profile_wgrad =
        get_env_bool_local("FTRAIN_HYGON_AG_GROUP_GEMM_WGRAD_PROFILE", false);
    hipEvent_t profile_start = nullptr;
    hipEvent_t profile_after_compact = nullptr;
    hipEvent_t profile_after_gemm = nullptr;
    if (profile_wgrad)
    {
        CUDA_CHECK(hipEventCreate(&profile_start));
        CUDA_CHECK(hipEventCreate(&profile_after_compact));
        CUDA_CHECK(hipEventCreate(&profile_after_gemm));
        CUDA_CHECK(hipEventRecord(profile_start, stream));
    }

    hipLaunchKernelGGL(ag_group_gemm_bf16_wgrad_compact_inputs_kernel,
                       dim3(sorted_capacity),
                       dim3(256),
                       0,
                       stream,
                       static_cast<const uint16_t *>(full_routed_input),
                       static_cast<const uint16_t *>(gathered_grad_output),
                       sorted_token_ids,
                       sorted_gather_index,
                       static_cast<uint16_t *>(compact_routed_input),
                       static_cast<uint16_t *>(compact_grad_output),
                       sorted_capacity,
                       full_m,
                       topk,
                       output_dim,
                       reduction_dim);
    CUDA_CHECK(hipGetLastError());
    if (profile_wgrad)
    {
        CUDA_CHECK(hipEventRecord(profile_after_compact, stream));
    }

    auto launch_output_slice = [&](int output_k_start,
                                   int output_k_extent) {
        hipLaunchKernelGGL(
            ag_group_gemm_bf16_wgrad_build_grouped_asm_args_kernel,
            dim3(num_experts),
            dim3(64),
            0,
            stream,
            static_cast<const uint16_t *>(compact_routed_input),
            static_cast<const uint16_t *>(compact_grad_output),
            global_expert_offsets,
            rank_cursors,
            static_cast<uint16_t *>(grad_weight),
            static_cast<uint8_t *>(grouped_asm_args),
            output_dim,
            reduction_dim,
            num_experts,
            world_size,
            output_k_start,
            output_k_extent);
        CUDA_CHECK(hipGetLastError());
        ag_wgrad_launch_grouped_asm(grouped_asm_args,
                                    output_k_extent,
                                    reduction_dim,
                                    num_experts,
                                    stream);
    };

    if (output_chunk_ready_flags == nullptr)
    {
        FTRAIN_HYGON_CHECK(output_chunk_size == 0)
            << "output_chunk_size requires output_chunk_ready_flags";
        launch_output_slice(0, output_dim);
    }
    else
    {
        FTRAIN_HYGON_CHECK(output_chunk_size > 0)
            << "FC1 wgrad output chunk size must be positive, got "
            << output_chunk_size;
        const int output_groups =
            (output_dim + output_chunk_size - 1) / output_chunk_size;
        for (int group = 0; group < output_groups; ++group)
        {
            // A stream-value wait does not occupy a resident CTA while the FC1
            // dgrad GEMM is producing this hidden/N chunk.  Launching a grid that
            // busy-waits in-kernel can consume all CUs and deadlock its producer.
            CUDA_CHECK(hipStreamWaitValue32(
                stream,
                const_cast<int *>(output_chunk_ready_flags + group),
                1u,
                hipStreamWaitValueGte,
                0xffffffffu));
            const int output_k_start = group * output_chunk_size;
            launch_output_slice(
                output_k_start,
                std::min(output_chunk_size, output_dim - output_k_start));
        }
    }

    if (profile_wgrad)
    {
        CUDA_CHECK(hipEventRecord(profile_after_gemm, stream));
        CUDA_CHECK(hipEventSynchronize(profile_after_gemm));
        float compact_ms = 0.0f;
        float gemm_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_CHECK(hipEventElapsedTime(
            &compact_ms, profile_start, profile_after_compact));
        CUDA_CHECK(hipEventElapsedTime(
            &gemm_ms, profile_after_compact, profile_after_gemm));
        CUDA_CHECK(hipEventElapsedTime(
            &total_ms, profile_start, profile_after_gemm));
        FTRAIN_HYGON_LOG_INFO("FC1 wgrad grouped-ASM profile compact_ms=%.6f "
                      "gemm_ms=%.6f total_ms=%.6f",
                      compact_ms,
                      gemm_ms,
                      total_ms);
        std::fprintf(stderr,
                     "[Hygon FC1 wgrad grouped-ASM profile] compact_ms=%.6f "
                     "gemm_ms=%.6f total_ms=%.6f\n",
                     compact_ms,
                     gemm_ms,
                     total_ms);
        CUDA_CHECK(hipEventDestroy(profile_start));
        CUDA_CHECK(hipEventDestroy(profile_after_compact));
        CUDA_CHECK(hipEventDestroy(profile_after_gemm));
    }
}

void ag_group_gemm_wait_rank_signals_impl(const int *rank_signals,
                                          int world_size,
                                          int signal_value,
                                          hipStream_t stream)
{
    hipLaunchKernelGGL(wait_rank_signals_kernel,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       rank_signals,
                       world_size,
                       signal_value);
}

void ag_group_gemm_reference_forward_impl(const void *full_input,
                                          const void *weight,
                                          const void *topk_ids,
                                          void *output,
                                          int full_m,
                                          int n_dim,
                                          int k_dim,
                                          int num_experts,
                                          int topk,
                                          c10::ScalarType dtype,
                                          c10::ScalarType index_dtype,
                                          hipStream_t stream)
{
    if (dtype == c10::ScalarType::BFloat16)
    {
        launch_reference_forward<__hip_bfloat16>(full_input,
                                                 weight,
                                                 topk_ids,
                                                 output,
                                                 full_m,
                                                 n_dim,
                                                 k_dim,
                                                 num_experts,
                                                 topk,
                                                 index_dtype,
                                                 stream);
    }
    else if (dtype == c10::ScalarType::Half)
    {
        launch_reference_forward<__half>(full_input,
                                         weight,
                                         topk_ids,
                                         output,
                                         full_m,
                                         n_dim,
                                         k_dim,
                                         num_experts,
                                         topk,
                                         index_dtype,
                                         stream);
    }
    else
    {
        FTRAIN_HYGON_CHECK(false) << "AGGroupGemm reference path supports FP16/BF16";
    }
}

struct AgPayloadDedupMix
{
    unsigned long long a;
    unsigned long long b;
};

__device__ inline AgPayloadDedupMix ag_payload_row_mix(
    const uint4 *__restrict__ row_vectors,
    int vector_count,
    int thread_stride)
{
    unsigned long long a = 0x9e3779b97f4a7c15ULL;
    unsigned long long b = 0xc2b2ae3d27d4eb4fULL;
    for (int index = threadIdx.x; index < vector_count;
         index += thread_stride)
    {
        const uint4 value = row_vectors[index];
        const unsigned int lanes[4] = {value.x, value.y, value.z, value.w};
#pragma unroll
        for (int lane = 0; lane < 4; ++lane)
        {
            const unsigned long long word =
                static_cast<unsigned long long>(lanes[lane]) +
                static_cast<unsigned long long>(index * 4 + lane) *
                    0x100000001b3ULL;
            a ^= word;
            a *= 0xff51afd7ed558ccdULL;
            a ^= a >> 29;
            b += word * 0xc4ceb9fe1a85ec53ULL;
            b ^= b >> 31;
            b *= 0x27d4eb2f165667c5ULL;
        }
    }
    return {a, b};
}

__global__ void ag_payload_signature_kernel(
    const uint4 *__restrict__ input,
    unsigned long long *__restrict__ signature_a,
    unsigned long long *__restrict__ signature_b,
    int rows,
    int vector_count)
{
    __shared__ unsigned long long shared_a[256];
    __shared__ unsigned long long shared_b[256];
    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }
    const AgPayloadDedupMix mix = ag_payload_row_mix(
        input + static_cast<int64_t>(row) * vector_count,
        vector_count,
        blockDim.x);
    shared_a[threadIdx.x] = mix.a;
    shared_b[threadIdx.x] = mix.b;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned long long a = mix.a;
        unsigned long long b = mix.b;
        for (int thread = 1; thread < blockDim.x; ++thread)
        {
            a ^= shared_a[thread] + 0x9e3779b97f4a7c15ULL;
            a *= 0xff51afd7ed558ccdULL;
            b += shared_b[thread] * 0xc4ceb9fe1a85ec53ULL;
            b ^= b >> 27;
        }
        signature_a[row] = a;
        signature_b[row] = b;
    }
}


__global__ void ag_payload_dedup_verify_kernel(
    const uint4 *__restrict__ input,
    const uint4 *__restrict__ compact_input,
    const int64_t *__restrict__ expanded_to_compact,
    int *__restrict__ mismatch_count,
    int rows,
    int topk,
    int vector_count)
{
    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }
    const int64_t mapped = expanded_to_compact[row];
    const uint4 *__restrict__ source =
        input + static_cast<int64_t>(row) * vector_count;
    const uint4 *__restrict__ representative =
        compact_input +
        static_cast<int64_t>(mapped / topk) * vector_count;
    for (int index = threadIdx.x; index < vector_count;
         index += blockDim.x)
    {
        const uint4 left = source[index];
        const uint4 right = representative[index];
        if (left.x != right.x || left.y != right.y ||
            left.z != right.z || left.w != right.w)
        {
            atomicAdd(mismatch_count, 1);
            return;
        }
    }
}

std::vector<at::Tensor> ag_payload_dedup_map_impl(
    const at::Tensor &input,
    int topk,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(input.is_cuda() && input.is_contiguous());
    FTRAIN_HYGON_CHECK(input.scalar_type() == c10::ScalarType::BFloat16);
    FTRAIN_HYGON_CHECK(input.dim() == 2);
    const int64_t rows = input.size(0);
    const int64_t k_dim = input.size(1);
    FTRAIN_HYGON_CHECK(rows > 0 && topk > 0 && rows % topk == 0);
    FTRAIN_HYGON_CHECK(k_dim > 0 && k_dim % 8 == 0)
        << "payload dedup requires K to be divisible by eight";
    const int compact_rows = static_cast<int>(rows / topk);
    auto options = input.options();
    auto long_options = options.dtype(at::kLong);
    auto ull_options = options.dtype(at::kUInt64);
    auto compact_input =
        at::empty({compact_rows, k_dim}, options);
    auto expanded_to_compact = at::empty({rows}, long_options);
    auto signature_a = at::empty({rows}, ull_options);
    auto signature_b = at::empty({rows}, ull_options);
    auto mismatch_count = at::zeros({1}, options.dtype(at::kInt));

    hipLaunchKernelGGL(ag_payload_signature_kernel,
                       dim3(static_cast<unsigned int>(rows)),
                       dim3(256),
                       0,
                       stream,
                       reinterpret_cast<const uint4 *>(input.data_ptr()),
                       reinterpret_cast<unsigned long long *>(
                           signature_a.data_ptr()),
                       reinterpret_cast<unsigned long long *>(
                           signature_b.data_ptr()),
                       static_cast<int>(rows),
                       static_cast<int>(k_dim / 8));
    CUDA_CHECK(hipGetLastError());

    // Stable two-pass sort gives a deterministic lexicographic signature
    // order without spin locks or a device-side hash table.
    auto signature_a_long = signature_a.view(at::kLong);
    auto signature_b_long = signature_b.view(at::kLong);
    auto order_b = at::argsort(signature_b_long, 0, true);
    auto signature_a_after_b = signature_a_long.index_select(0, order_b);
    auto signature_b_after_b = signature_b_long.index_select(0, order_b);
    auto sorted_a_pair = at::sort(signature_a_after_b, 0, true);
    auto sorted_a = std::get<0>(sorted_a_pair);
    auto order_a = std::get<1>(sorted_a_pair);
    auto sorted_rows = order_b.index_select(0, order_a);
    auto sorted_b = signature_b_after_b.index_select(0, order_a);
    auto changed_a = at::narrow(sorted_a, 0, 1, rows - 1) !=
                     at::narrow(sorted_a, 0, 0, rows - 1);
    auto changed_b = at::narrow(sorted_b, 0, 1, rows - 1) !=
                     at::narrow(sorted_b, 0, 0, rows - 1);
    auto boundaries = (changed_a | changed_b).to(at::kLong);
    auto group_sorted = at::cat(
        {at::zeros({1}, signature_a.options().dtype(at::kLong)),
         at::cumsum(boundaries, 0)},
        0);
    auto unique_count = at::narrow(group_sorted, 0, rows - 1, 1) + 1;
    auto counts = at::bincount(
        group_sorted, at::Tensor(), compact_rows);
    auto offsets = at::cat(
        {at::zeros({1}, signature_a.options().dtype(at::kLong)),
         at::cumsum(counts, 0).slice(0, 0, compact_rows - 1)},
        0);
    auto occurrence_sorted = at::arange(
        rows, signature_a.options().dtype(at::kLong)) -
        offsets.index_select(0, group_sorted);
    expanded_to_compact.index_put_(
        {sorted_rows}, group_sorted * topk + occurrence_sorted);
    auto first_rows = sorted_rows.index_select(0, offsets);
    compact_input.copy_(input.index_select(0, first_rows));

    if (!get_env_bool_with_legacy(
            "FTRAIN_HYGON_PAYLOAD_DEDUP_SKIP_VERIFY",
            "FTRAIN_HYGON_PAYLOAD_DEDUP_SKIP_VERIFY",
            false))
    {
        hipLaunchKernelGGL(ag_payload_dedup_verify_kernel,
                           dim3(static_cast<unsigned int>(rows)),
                           dim3(256),
                           0,
                           stream,
                           reinterpret_cast<const uint4 *>(input.data_ptr()),
                           reinterpret_cast<const uint4 *>(
                               compact_input.data_ptr()),
                           expanded_to_compact.data_ptr<int64_t>(),
                           mismatch_count.data_ptr<int>(),
                           static_cast<int>(rows),
                           topk,
                           static_cast<int>(k_dim / 8));
        CUDA_CHECK(hipGetLastError());
    }

    // Keep this helper stream-ordered.  Callers that need a hard correctness
    // gate can inspect the returned counters after a synchronization.
    return {
        compact_input,
        expanded_to_compact,
        unique_count,
        mismatch_count,
    };
}

__global__ void ag_payload_output_remap_kernel(
    const uint4 *__restrict__ source,
    const int64_t *__restrict__ row_map,
    uint4 *__restrict__ destination,
    int rows,
    int vector_count)
{
    const int row = blockIdx.x;
    if (row >= rows)
    {
        return;
    }
    const int64_t source_row = row_map[row];
    const uint4 *__restrict__ source_row_ptr =
        source + source_row * static_cast<int64_t>(vector_count);
    uint4 *__restrict__ destination_row_ptr =
        destination + static_cast<int64_t>(row) * vector_count;
    for (int index = threadIdx.x; index < vector_count;
         index += blockDim.x)
    {
        destination_row_ptr[index] = source_row_ptr[index];
    }
}

__global__ void ag_payload_scatter_topk_kernel(
    const uint4 *__restrict__ source,
    const int64_t *__restrict__ destination_rows,
    uint4 *__restrict__ destination,
    int source_rows,
    int topk,
    int vector_count)
{
    const int token = blockIdx.x;
    if (token >= source_rows)
    {
        return;
    }
    const uint4 *__restrict__ source_row_ptr =
        source + static_cast<int64_t>(token) * vector_count;
    for (int index = threadIdx.x; index < vector_count;
         index += blockDim.x)
    {
        const uint4 value = source_row_ptr[index];
#pragma unroll 2
        for (int replica = 0; replica < topk; ++replica)
        {
            const int64_t destination_row =
                destination_rows[static_cast<int64_t>(token) * topk + replica];
            destination[destination_row * vector_count + index] = value;
        }
    }
}

at::Tensor ag_payload_scatter_topk_impl(
    const at::Tensor &source,
    const at::Tensor &destination_rows,
    at::Tensor destination,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(source.is_cuda() && source.is_contiguous());
    FTRAIN_HYGON_CHECK(destination_rows.is_cuda() && destination_rows.is_contiguous());
    FTRAIN_HYGON_CHECK(destination.is_cuda() && destination.is_contiguous());
    FTRAIN_HYGON_CHECK(source.scalar_type() == c10::ScalarType::BFloat16);
    FTRAIN_HYGON_CHECK(destination.scalar_type() == c10::ScalarType::BFloat16);
    FTRAIN_HYGON_CHECK(destination_rows.scalar_type() == c10::ScalarType::Long);
    FTRAIN_HYGON_CHECK(source.dim() == 2 && destination.dim() == 2);
    FTRAIN_HYGON_CHECK(destination_rows.dim() == 2);
    FTRAIN_HYGON_CHECK(destination_rows.size(0) == source.size(0));
    FTRAIN_HYGON_CHECK(destination_rows.size(1) > 0);
    FTRAIN_HYGON_CHECK(source.size(1) == destination.size(1));
    const int64_t source_rows = source.size(0);
    const int64_t n_dim = source.size(1);
    FTRAIN_HYGON_CHECK(n_dim > 0 && n_dim % 8 == 0);
    FTRAIN_HYGON_CHECK(source_rows > 0 && source_rows <= 0x7fffffff);

    hipLaunchKernelGGL(ag_payload_scatter_topk_kernel,
                       dim3(static_cast<unsigned int>(source_rows)),
                       dim3(1024),
                       0,
                       stream,
                       reinterpret_cast<const uint4 *>(source.data_ptr()),
                       destination_rows.data_ptr<int64_t>(),
                       reinterpret_cast<uint4 *>(destination.data_ptr()),
                       static_cast<int>(source_rows),
                       static_cast<int>(destination_rows.size(1)),
                       static_cast<int>(n_dim / 8));
    CUDA_CHECK(hipGetLastError());
    return destination;
}

at::Tensor ag_payload_output_remap_impl(
    const at::Tensor &source,
    const at::Tensor &row_map,
    at::Tensor destination,
    hipStream_t stream)
{
    FTRAIN_HYGON_CHECK(source.is_cuda() && source.is_contiguous());
    FTRAIN_HYGON_CHECK(destination.is_cuda() && destination.is_contiguous());
    FTRAIN_HYGON_CHECK(row_map.is_cuda() && row_map.is_contiguous());
    FTRAIN_HYGON_CHECK(source.scalar_type() == c10::ScalarType::BFloat16);
    FTRAIN_HYGON_CHECK(destination.scalar_type() == c10::ScalarType::BFloat16);
    FTRAIN_HYGON_CHECK(row_map.scalar_type() == c10::ScalarType::Long);
    FTRAIN_HYGON_CHECK(source.dim() == 2 && destination.dim() == 2);
    FTRAIN_HYGON_CHECK(row_map.dim() == 1 && row_map.numel() == destination.size(0));
    FTRAIN_HYGON_CHECK(source.size(1) == destination.size(1));
    // Source may have fewer rows than destination when gathering duplicated
    // compact payload into an expanded layout.
    const int64_t rows = destination.size(0);
    const int64_t n_dim = source.size(1);
    FTRAIN_HYGON_CHECK(n_dim > 0 && n_dim % 8 == 0)
        << "payload output remap requires N to be divisible by eight";
    FTRAIN_HYGON_CHECK(rows > 0 && rows <= 0x7fffffff);

    const int block_threads = std::min(
        1024,
        std::max(
            64,
            get_env_int_with_legacy(
                "FTRAIN_HYGON_PAYLOAD_REMAP_BLOCK_THREADS",
                "FTRAIN_HYGON_PAYLOAD_REMAP_BLOCK_THREADS",
                1024)));
    hipLaunchKernelGGL(ag_payload_output_remap_kernel,
                       dim3(static_cast<unsigned int>(rows)),
                       dim3(block_threads),
                       0,
                       stream,
                       reinterpret_cast<const uint4 *>(source.data_ptr()),
                       row_map.data_ptr<int64_t>(),
                       reinterpret_cast<uint4 *>(destination.data_ptr()),
                       static_cast<int>(rows),
                       static_cast<int>(n_dim / 8));
    CUDA_CHECK(hipGetLastError());
    return destination;
}

} // namespace hygon_vendor
} // namespace bytedance
