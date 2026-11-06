#include "hygon_vendor.h"
#include "hip_kernels/helper_kernels.h"
#include "hip/hip_runtime.h"

namespace bytedance
{
namespace hygon_vendor
{

using half4_t = __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
using half8_t = __attribute__((__vector_size__(8 * sizeof(_Float16)))) _Float16;
using byte16_t = __attribute__((__vector_size__(16 * sizeof(char)))) char;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using floatx4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;


__launch_bounds__(1024, 1) __global__
    void custom_copy_all_gather_kernel(int vec_size,
                                       int world_size,
                                       int rank,
                                       int rows,
                                       int cols,
                                       void **pp_srcs,
                                       int *barrier_ptr)
{
    // 线程号
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 每个线程处理的起始元素索引
    int start_index = tid * vec_size;
    // 线程总数
    int thread_num = gridDim.x * blockDim.x;
    // 网格跨步循环大小
    int stride = thread_num * vec_size;
    // rank维度循环 处理的元素个数
    int chunk_size = rows * cols / world_size;
    // 可以被vec_size整除的元素边界
    int vector_bound = (chunk_size / vec_size) * vec_size;

    for (int i = rank + 1; i < (world_size + rank); ++i)
    {
        int id = i % world_size;
        _Float16 *p_from =
            static_cast<_Float16 *>(pp_srcs[id]) + id * chunk_size;
        _Float16 *p_to =
            static_cast<_Float16 *>(pp_srcs[rank]) + id * chunk_size;

        // 向量化处理
        for (int j = start_index; j < vector_bound; j += stride)
        {
            half8_t *src_data = reinterpret_cast<half8_t *>(p_from + j);
            half8_t *dst_data = reinterpret_cast<half8_t *>(p_to + j);
            *dst_data = *src_data;
        }
        // 尾处理
        int scalar_start = vector_bound + tid;
        int scalar_stride = thread_num;
        for (int j = scalar_start; j < chunk_size; j += scalar_stride)
        {
            p_to[j] = p_from[j];
        }

        // 设置barrier标志,原子加实现块间同步
        asm volatile("s_barrier\n\t");
        if (0 == threadIdx.x)
        {
            int value = atomicAdd_system(barrier_ptr + id + world_size, 1);
            // 处理当前chunk数据的所有块都完成数据通信，将标志置1
            if ((gridDim.x - 1) == value)
            {
                barrier_ptr[id] = 1;
                barrier_ptr[id + world_size] = 0;
            }
        }
    }
}

__launch_bounds__(1024, 1) __global__
    void custom_copy_all_gather_chunk_kernel(int vec_size,
                                             int world_size,
                                             int rank,
                                             int rows,
                                             int cols,
                                             int chunk_start,
                                             int chunk_rows,
                                             void **pp_srcs)
{
    int local_rows = rows / world_size;
    if (chunk_rows <= 0 || local_rows <= 0)
    {
        return;
    }

    int src_rank = chunk_start / local_rows;
    src_rank = src_rank < 0 ? 0 : src_rank;
    src_rank = src_rank >= world_size ? world_size - 1 : src_rank;
    const int rows_left = rows - chunk_start;
    int rows_in_chunk = rows_left < chunk_rows ? rows_left : chunk_rows;
    const int rank_row = chunk_start - src_rank * local_rows;
    if (rows_in_chunk > local_rows - rank_row)
    {
        rows_in_chunk = local_rows - rank_row;
    }
    if (rows_in_chunk <= 0)
    {
        return;
    }

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = gridDim.x * blockDim.x;
    int start_index = tid * vec_size;
    int stride = thread_num * vec_size;
    int elem_count = rows_in_chunk * cols;
    int vector_bound = (elem_count / vec_size) * vec_size;

    _Float16 *p_from =
        static_cast<_Float16 *>(pp_srcs[src_rank]) + chunk_start * cols;
    _Float16 *p_to =
        static_cast<_Float16 *>(pp_srcs[rank]) + chunk_start * cols;

    for (int j = start_index; j < vector_bound; j += stride)
    {
        half8_t *src_data = reinterpret_cast<half8_t *>(p_from + j);
        half8_t *dst_data = reinterpret_cast<half8_t *>(p_to + j);
        *dst_data = *src_data;
    }

    int scalar_start = vector_bound + tid;
    for (int j = scalar_start; j < elem_count; j += thread_num)
    {
        p_to[j] = p_from[j];
    }
}

__launch_bounds__(1024, 1) __global__
    void custom_copy_all_gather_chunks_signal_kernel(int vec_size,
                                                     int world_size,
                                                     int rank,
                                                     int rows,
                                                     int cols,
                                                     int chunk_rows,
                                                     int signal_value,
                                                     int *barrier_ptr,
                                                     void **pp_srcs,
                                                     bool skip_local_rank,
                                                     int stage_begin,
                                                     int stage_end)
{
    const int local_rows = rows / world_size;
    if (chunk_rows <= 0 || local_rows <= 0)
    {
        return;
    }

    const int chunks_per_rank = (local_rows + chunk_rows - 1) / chunk_rows;
    const int total_chunks = chunks_per_rank * world_size;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_num = gridDim.x * blockDim.x;

    const int first_stage = stage_begin < 0 ? 0 : stage_begin;
    const int last_stage = stage_end > world_size ? world_size : stage_end;
    for (int stage = first_stage; stage < last_stage; ++stage)
    {
        const int src_rank = (rank + stage) % world_size;
        if (skip_local_rank && src_rank == rank)
        {
            continue;
        }
        for (int rank_chunk = 0; rank_chunk < chunks_per_rank; ++rank_chunk)
        {
            const int chunk_id = src_rank * chunks_per_rank + rank_chunk;
            const int row_start =
                src_rank * local_rows + rank_chunk * chunk_rows;
            int rows_in_chunk = local_rows - rank_chunk * chunk_rows;
            rows_in_chunk =
                rows_in_chunk < chunk_rows ? rows_in_chunk : chunk_rows;
            if (rows_in_chunk <= 0)
            {
                continue;
            }

            const int elem_count = rows_in_chunk * cols;
            const int vector_bound = (elem_count / vec_size) * vec_size;
            const int start_index = tid * vec_size;
            const int stride = thread_num * vec_size;

            _Float16 *p_from =
                static_cast<_Float16 *>(pp_srcs[src_rank]) + row_start * cols;
            _Float16 *p_to =
                static_cast<_Float16 *>(pp_srcs[rank]) + row_start * cols;

            for (int j = start_index; j < vector_bound; j += stride)
            {
                half8_t *src_data = reinterpret_cast<half8_t *>(p_from + j);
                half8_t *dst_data = reinterpret_cast<half8_t *>(p_to + j);
                *dst_data = *src_data;
            }

            for (int j = vector_bound + tid; j < elem_count; j += thread_num)
            {
                p_to[j] = p_from[j];
            }

            asm volatile("s_barrier\n\t");
            if (threadIdx.x == 0)
            {
                const int counter_index = total_chunks + chunk_id;
                const int value =
                    atomicAdd_system(barrier_ptr + counter_index, 1);
                if ((gridDim.x - 1) == value)
                {
                    // Peer-visible chunk readiness must be a system-scope
                    // release.  A normal global store after the DMA can be
                    // observed early by a hand-written MFMA consumer.
                    __hip_atomic_store(barrier_ptr + chunk_id,
                                       signal_value,
                                       __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
                    barrier_ptr[counter_index] = 0;
                }
            }
            asm volatile("s_barrier\n\t");
        }
    }
}

__launch_bounds__(1024, 1) __global__
    void signal_local_chunks_kernel(int world_size,
                                    int rank,
                                    int rows,
                                    int chunk_rows,
                                    int signal_value,
                                    int *barrier_ptr)
{
    const int local_rows = rows / world_size;
    if (chunk_rows <= 0 || local_rows <= 0)
    {
        return;
    }
    const int chunks_per_rank = (local_rows + chunk_rows - 1) / chunk_rows;
    for (int rank_chunk = threadIdx.x; rank_chunk < chunks_per_rank;
         rank_chunk += blockDim.x)
    {
        barrier_ptr[rank * chunks_per_rank + rank_chunk] = signal_value;
    }
}

// V40 true-ring transport with chunk-granular readiness.  One kernel instance
// owns all seven forwarding stages for this rank.  Stage 1 publishes the local
// expanded shard; later stages wait for an incoming chunk signal before
// forwarding that same chunk to the previous rank.  The signal layout is the
// physical owner-rank/chunk layout consumed by the variant-17 hybrid scheduler.
__launch_bounds__(1024, 1) __global__
    void custom_copy_ring_b16_chunks_signal_kernel(int vec_size,
                                                   int world_size,
                                                   int rank,
                                                   int rows,
                                                   int cols,
                                                   int chunk_rows,
                                                   int signal_value,
                                                   int *local_signal,
                                                   int *remote_signal,
                                                   const _Float16 *stage1_src,
                                                   void **pp_srcs)
{
    const int local_rows = rows / world_size;
    if (chunk_rows <= 0 || local_rows <= 0 || cols <= 0)
    {
        return;
    }

    const int chunks_per_rank =
        (local_rows + chunk_rows - 1) / chunk_rows;
    const int total_chunks = chunks_per_rank * world_size;
    const int dst_rank = (rank - 1 + world_size) % world_size;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int thread_num = gridDim.x * blockDim.x;
    _Float16 *dst_base = static_cast<_Float16 *>(pp_srcs[dst_rank]);
    _Float16 *local_base = static_cast<_Float16 *>(pp_srcs[rank]);

    for (int stage = 1; stage < world_size; ++stage)
    {
        const int owner_rank = (rank + stage - 1) % world_size;
        for (int rank_chunk = 0; rank_chunk < chunks_per_rank; ++rank_chunk)
        {
            const int chunk_id =
                owner_rank * chunks_per_rank + rank_chunk;
            if (stage > 1)
            {
                if (threadIdx.x == 0)
                {
                    while (__hip_atomic_load(local_signal + chunk_id,
                                             __ATOMIC_ACQUIRE,
                                             __HIP_MEMORY_SCOPE_SYSTEM) <
                           signal_value)
                    {
                        __builtin_amdgcn_s_sleep(1);
                    }
                }
                __syncthreads();
            }

            const int row_in_chunk = rank_chunk * chunk_rows;
            int rows_in_chunk = local_rows - row_in_chunk;
            rows_in_chunk =
                rows_in_chunk < chunk_rows ? rows_in_chunk : chunk_rows;
            if (rows_in_chunk <= 0)
            {
                continue;
            }

            const int64_t chunk_elem =
                static_cast<int64_t>(rows_in_chunk) * cols;
            const int64_t vector_bound =
                (chunk_elem / vec_size) * vec_size;
            const int64_t start_index = tid * vec_size;
            const int64_t stride =
                static_cast<int64_t>(thread_num) * vec_size;
            const int64_t chunk_elem_start =
                static_cast<int64_t>(owner_rank * local_rows + row_in_chunk) *
                cols;

            const _Float16 *src = stage == 1
                ? stage1_src +
                      static_cast<int64_t>(row_in_chunk) * cols
                : local_base + chunk_elem_start;
            _Float16 *dst = dst_base + chunk_elem_start;

            for (int64_t j = start_index; j < vector_bound; j += stride)
            {
                const half8_t *src_data =
                    reinterpret_cast<const half8_t *>(src + j);
                half8_t *dst_data =
                    reinterpret_cast<half8_t *>(dst + j);
                *dst_data = *src_data;
            }

            for (int64_t j = vector_bound + tid; j < chunk_elem;
                 j += thread_num)
            {
                dst[j] = src[j];
            }

            // Make every thread's peer payload store system-visible before the
            // block participates in the completion counter.  A block barrier
            // alone is not a system-scope release for cross-GPU consumers.
            __threadfence_system();
            __syncthreads();
            if (threadIdx.x == 0)
            {
                // Keep completion counters in this rank's local signal scratch.
                // The remote buffer only receives the final release word, so
                // a peer never needs write access to another rank's counters.
                const int counter_index = total_chunks + chunk_id;
                const int value =
                    atomicAdd_system(local_signal + counter_index, 1);
                if ((gridDim.x - 1) == value)
                {
                    __hip_atomic_store(remote_signal + chunk_id,
                                       signal_value,
                                       __ATOMIC_RELEASE,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
                    local_signal[counter_index] = 0;
                }
            }
            __syncthreads();
        }
    }
}

__launch_bounds__(1024, 1) __global__
    void custom_copy_all_gather_s8_kernel(int vec_size,
                                          int world_size,
                                          int rank,
                                          int rows,
                                          int cols,
                                          void **pp_srcs,
                                          void **pp_scales,
                                          int *barrier_ptr)
{
    constexpr int SCALE_VEC = 4;
    // 线程号
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 每个线程处理的起始元素索引
    int start_index = tid * vec_size;
    int scale_start_index = tid * SCALE_VEC;
    // 线程总数
    int thread_num = gridDim.x * blockDim.x;
    // 网格跨步循环大小
    int stride = thread_num * vec_size;
    int scale_stride = thread_num * SCALE_VEC;
    // rank维度循环 处理的元素个数
    int chunk_size = rows * cols / world_size;
    int scale_chunk_size = rows / world_size;
    // 可以被vec_size整除的元素边界
    int vector_bound = (chunk_size / vec_size) * vec_size;
    int scale_vector_bound = (scale_chunk_size / SCALE_VEC) * SCALE_VEC;

    for (int i = rank + 1; i < (world_size + rank); ++i)
    {
        int id = i % world_size;
        // inputs
        char *p_from = static_cast<char *>(pp_srcs[id]) + id * chunk_size;
        char *p_to = static_cast<char *>(pp_srcs[rank]) + id * chunk_size;
        // 向量化处理
        for (int j = start_index; j < vector_bound; j += stride)
        {
            byte16_t *src_data = reinterpret_cast<byte16_t *>(p_from + j);
            byte16_t *dst_data = reinterpret_cast<byte16_t *>(p_to + j);
            *dst_data = *src_data;
        }
        // 尾处理
        int scalar_start = vector_bound + tid;
        int scalar_stride = thread_num;
        for (int j = scalar_start; j < chunk_size; j += scalar_stride)
        {
            p_to[j] = p_from[j];
        }

        // scales
        float *p_scale_from =
            static_cast<float *>(pp_scales[id]) + id * scale_chunk_size;
        float *p_scale_to =
            static_cast<float *>(pp_scales[rank]) + id * scale_chunk_size;
        // 向量化处理
        for (int j = scale_start_index; j < scale_vector_bound;
             j += scale_stride)
        {
            floatx4_t *src_data =
                reinterpret_cast<floatx4_t *>(p_scale_from + j);
            floatx4_t *dst_data = reinterpret_cast<floatx4_t *>(p_scale_to + j);
            *dst_data = *src_data;
        }
        // 尾处理
        scalar_start = scale_vector_bound + tid;
        scalar_stride = thread_num;
        for (int j = scalar_start; j < scale_chunk_size; j += scalar_stride)
        {
            p_scale_to[j] = p_scale_from[j];
        }

        // 设置barrier标志,原子加实现块间同步
        asm volatile("s_barrier\n\t");
        if (0 == threadIdx.x)
        {
            int value = atomicAdd_system(barrier_ptr + id + world_size, 1);
            // 处理当前chunk数据的所有块都完成数据通信，将标志置1
            if ((gridDim.x - 1) == value)
            {
                barrier_ptr[id] = 1;
                barrier_ptr[id + world_size] = 0;
            }
        }
    }
}

__launch_bounds__(1024, 1) __global__
    void custom_copy_all_gather_fp8_kernel(int vec_size,
                                           int world_size,
                                           int rank,
                                           int rows,
                                           int cols,
                                           void **pp_srcs,
                                           void **pp_scales,
                                           int *barrier_ptr,
                                           bool all_gather_scale_a)
{
    constexpr int SCALE_VEC = 4;
    // 线程号
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 每个线程处理的起始元素索引
    int start_index = tid * vec_size;
    int scale_start_index = tid * SCALE_VEC;
    // 线程总数
    int thread_num = gridDim.x * blockDim.x;
    // 网格跨步循环大小
    int stride = thread_num * vec_size;
    int scale_stride = thread_num * SCALE_VEC;
    // rank维度循环 处理的元素个数
    int chunk_size = rows * cols / world_size;
    int scale_chunk_size = rows / world_size;
    // 可以被vec_size整除的元素边界
    int vector_bound = (chunk_size / vec_size) * vec_size;
    int scale_vector_bound = (scale_chunk_size / SCALE_VEC) * SCALE_VEC;

    for (int i = rank + 1; i < (world_size + rank); ++i)
    {
        int id = i % world_size;
        // inputs
        char *p_from = static_cast<char *>(pp_srcs[id]) + id * chunk_size;
        char *p_to = static_cast<char *>(pp_srcs[rank]) + id * chunk_size;
        // 向量化处理
        for (int j = start_index; j < vector_bound; j += stride)
        {
            byte16_t *src_data = reinterpret_cast<byte16_t *>(p_from + j);
            byte16_t *dst_data = reinterpret_cast<byte16_t *>(p_to + j);
            *dst_data = *src_data;
        }
        // 尾处理
        int scalar_start = vector_bound + tid;
        int scalar_stride = thread_num;
        for (int j = scalar_start; j < chunk_size; j += scalar_stride)
        {
            p_to[j] = p_from[j];
        }

        if (all_gather_scale_a)
        {
            // scales
            float *p_scale_from =
                static_cast<float *>(pp_scales[id]) + id * scale_chunk_size;
            float *p_scale_to =
                static_cast<float *>(pp_scales[rank]) + id * scale_chunk_size;
            // 向量化处理
            for (int j = scale_start_index; j < scale_vector_bound;
                 j += scale_stride)
            {
                floatx4_t *src_data =
                    reinterpret_cast<floatx4_t *>(p_scale_from + j);
                floatx4_t *dst_data =
                    reinterpret_cast<floatx4_t *>(p_scale_to + j);
                *dst_data = *src_data;
            }
            // 尾处理
            scalar_start = scale_vector_bound + tid;
            scalar_stride = thread_num;
            for (int j = scalar_start; j < scale_chunk_size; j += scalar_stride)
            {
                p_scale_to[j] = p_scale_from[j];
            }
        }

        // 设置barrier标志,原子加实现块间同步
        asm volatile("s_barrier\n\t");
        if (0 == threadIdx.x)
        {
            int value = atomicAdd_system(barrier_ptr + id + world_size, 1);
            // 处理当前chunk数据的所有块都完成数据通信，将标志置1
            if ((gridDim.x - 1) == value)
            {
                barrier_ptr[id] = 1;
                barrier_ptr[id + world_size] = 0;
            }
        }
    }
}


void ag_copy_remote_b16_impl(int grid_size,
                             int block_size,
                             int vec_size,
                             int world_size,
                             int rank,
                             int rows,
                             int cols,
                             void **pp_srcs,
                             int *barrier_ptr,
                             hipStream_t stream)
{
    custom_copy_all_gather_kernel<<<grid_size, block_size, 0, stream>>>(
        vec_size,
        world_size,
        rank,
        rows,
        cols,
        pp_srcs,
        barrier_ptr);
}

void ag_copy_remote_b16_chunk_impl(int grid_size,
                                   int block_size,
                                   int vec_size,
                                   int world_size,
                                   int rank,
                                   int rows,
                                   int cols,
                                   int chunk_start,
                                   int chunk_rows,
                                   void **pp_srcs,
                                   hipStream_t stream)
{
    custom_copy_all_gather_chunk_kernel<<<grid_size, block_size, 0, stream>>>(
        vec_size,
        world_size,
        rank,
        rows,
        cols,
        chunk_start,
        chunk_rows,
        pp_srcs);
}

void ag_copy_remote_b16_chunks_signal_impl(int grid_size,
                                           int block_size,
                                           int vec_size,
                                           int world_size,
                                           int rank,
                                           int rows,
                                           int cols,
                                           int chunk_rows,
                                           int signal_value,
                                           int *barrier_ptr,
                                           void **pp_srcs,
                                           hipStream_t stream)
{
    custom_copy_all_gather_chunks_signal_kernel<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(vec_size,
                                                            world_size,
                                                            rank,
                                                            rows,
                                                            cols,
                                                            chunk_rows,
                                                            signal_value,
                                                            barrier_ptr,
                                                            pp_srcs,
                                                            false,
                                                            0,
                                                            world_size);
}

void ag_copy_remote_b16_chunks_signal_impl(int grid_size,
                                           int block_size,
                                           int vec_size,
                                           int world_size,
                                           int rank,
                                           int rows,
                                           int cols,
                                           int chunk_rows,
                                           int *barrier_ptr,
                                           void **pp_srcs,
                                           hipStream_t stream)
{
    ag_copy_remote_b16_chunks_signal_impl(grid_size,
                                          block_size,
                                          vec_size,
                                          world_size,
                                          rank,
                                          rows,
                                          cols,
                                          chunk_rows,
                                          1,
                                          barrier_ptr,
                                          pp_srcs,
                                          stream);
}

void ag_signal_local_chunks_impl(int block_size,
                                 int world_size,
                                 int rank,
                                 int rows,
                                 int chunk_rows,
                                 int signal_value,
                                 int *barrier_ptr,
                                 hipStream_t stream)
{
    signal_local_chunks_kernel<<<1, block_size, 0, stream>>>(world_size,
                                                             rank,
                                                             rows,
                                                             chunk_rows,
                                                             signal_value,
                                                             barrier_ptr);
}

void ag_copy_remote_b16_chunks_signal_skip_local_impl(int grid_size,
                                                      int block_size,
                                                      int vec_size,
                                                      int world_size,
                                                      int rank,
                                                      int rows,
                                                      int cols,
                                                      int chunk_rows,
                                                      int signal_value,
                                                      int *barrier_ptr,
                                                      void **pp_srcs,
                                                      hipStream_t stream)
{
    custom_copy_all_gather_chunks_signal_kernel<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(vec_size,
                                                            world_size,
                                                            rank,
                                                            rows,
                                                            cols,
                                                            chunk_rows,
                                                            signal_value,
                                                            barrier_ptr,
                                                            pp_srcs,
                                                            true,
                                                            0,
                                                            world_size);
}

void ag_copy_remote_b16_chunks_signal_stages_impl(int grid_size,
                                                  int block_size,
                                                  int vec_size,
                                                  int world_size,
                                                  int rank,
                                                  int rows,
                                                  int cols,
                                                  int chunk_rows,
                                                  int signal_value,
                                                  int stage_begin,
                                                  int stage_end,
                                                  int *barrier_ptr,
                                                  void **pp_srcs,
                                                  hipStream_t stream)
{
    custom_copy_all_gather_chunks_signal_kernel<<<grid_size,
                                                  block_size,
                                                  0,
                                                  stream>>>(vec_size,
                                                            world_size,
                                                            rank,
                                                            rows,
                                                            cols,
                                                            chunk_rows,
                                                            signal_value,
                                                            barrier_ptr,
                                                            pp_srcs,
                                                            true,
                                                    stage_begin,
        stage_end);
}

void ag_copy_remote_b16_chunks_signal_stages_grouped_impl(
    int grid_size,
    int block_size,
    int vec_size,
    int world_size,
    int rank,
    int rows,
    int cols,
    int chunk_rows,
    int signal_value,
    int stage_begin,
    int stage_end,
    int *barrier_ptr,
    void **pp_srcs,
    hipStream_t stream,
    int group_chunks)
{
    (void)group_chunks;
    ag_copy_remote_b16_chunks_signal_stages_impl(
        grid_size,
        block_size,
        vec_size,
        world_size,
        rank,
        rows,
        cols,
        chunk_rows,
        signal_value,
        stage_begin,
        stage_end,
        barrier_ptr,
        pp_srcs,
        stream);
}

__launch_bounds__(1024, 1) __global__
    void ragged_ring_payload_acquire_kernel(
        const void *__restrict__ payload,
        size_t bytes,
        int *__restrict__ internal_signal,
        int *__restrict__ owner_signal,
        int *__restrict__ completion_counter,
        uint32_t signal_value,
        int probe_stride_words,
        int probe_mode)
{
    const uint32_t *words = reinterpret_cast<const uint32_t *>(payload);
    const size_t count = bytes / sizeof(uint32_t);
    const size_t stride_words = static_cast<size_t>(
        probe_stride_words > 0 ? probe_stride_words : 1);
    const size_t probe_count = (count + stride_words - 1) / stride_words;
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    uint32_t sink = 0;
    for (size_t item = static_cast<size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
         item < probe_count;
         item += stride)
    {
        const uint32_t *source = words + item * stride_words;
        if (probe_mode == 3)
        {
            // Lightweight latency/fence protocol: do not sweep payload lines.
            // The upstream hipStreamWaitValue32 has already observed the raw
            // owner signal; this kernel supplies a consumer-side execution and
            // system-fence hop before publishing the body-facing signal.
            sink ^= 1U;
        }
        else if (probe_mode == 1)
        {
            sink ^= *reinterpret_cast<const volatile uint32_t *>(source);
        }
        else if (probe_mode == 2)
        {
            sink ^= __builtin_nontemporal_load(source);
        }
        else
        {
            sink ^= __hip_atomic_load(source,
                                      __ATOMIC_RELAXED,
                                      __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
    // Keep the read alive and order it before the completion atomic.
    if (sink == 0xdeadbeefU)
    {
        *completion_counter = static_cast<int>(sink);
    }

    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const int arrived = atomicAdd_system(completion_counter, 1);
        if (arrived == static_cast<int>(gridDim.x) - 1)
        {
            __hip_atomic_store(owner_signal,
                               signal_value,
                               __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_SYSTEM);
            __hip_atomic_store(internal_signal,
                               signal_value,
                               __ATOMIC_RELEASE,
                               __HIP_MEMORY_SCOPE_SYSTEM);
            atomicExch_system(completion_counter, 0);
        }
    }
}

__launch_bounds__(64, 1) __global__
    void ragged_ring_light_acquire_group_kernel(
        int *__restrict__ local_signals,
        uint32_t signal_value,
        int owner_begin,
        int owner_count,
        int rank,
        int body_slot_base,
        int *__restrict__ ready_mask)
{
    const int owner = owner_begin + static_cast<int>(blockIdx.x);
    if (owner < owner_begin || owner >= owner_begin + owner_count ||
        owner == rank || threadIdx.x != 0)
    {
        return;
    }

    // The host enqueued hipStreamWaitValue32 for every owner in this group
    // before launching this kernel.  Preserve that launch boundary, then publish
    // all body-facing owner slots with consumer-side system fences.
    __threadfence_system();
    __hip_atomic_store(local_signals + body_slot_base + owner,
                       signal_value,
                       __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    if (ready_mask != nullptr)
    {
        atomicOr_system(ready_mask, 1U << owner);
    }
}

void ag_copy_ring_b16_chunks_signal_impl(int grid_size,
                                         int block_size,
                                         int vec_size,
                                         int world_size,
                                         int rank,
                                         int rows,
                                         int cols,
                                         int chunk_rows,
                                         int signal_value,
                                         int *local_signal,
                                         int *remote_signal,
                                         const void *stage1_src,
                                         void **pp_srcs,
                                         hipStream_t stream)
{
    custom_copy_ring_b16_chunks_signal_kernel<<<grid_size,
                                                block_size,
                                                0,
                                                stream>>>(
        vec_size,
        world_size,
        rank,
        rows,
        cols,
        chunk_rows,
        signal_value,
        local_signal,
        remote_signal,
        static_cast<const _Float16 *>(stage1_src),
        pp_srcs);
}

void ag_copy_remote_s8_impl(int grid_size,
                            int block_size,
                            int vec_size,
                            int world_size,
                            int rank,
                            int rows,
                            int cols,
                            void **pp_srcs,
                            void **pp_scales,
                            int *barrier_ptr,
                            hipStream_t stream,
                            bool all_gather_scale_a)
{
    custom_copy_all_gather_s8_kernel<<<grid_size, block_size, 0, stream>>>(
        vec_size,
        world_size,
        rank,
        rows,
        cols,
        pp_srcs,
        pp_scales,
        barrier_ptr);
}

void ag_copy_remote_fp8_impl(int grid_size,
                             int block_size,
                             int vec_size,
                             int world_size,
                             int rank,
                             int rows,
                             int cols,
                             void **pp_srcs,
                             void **pp_scales,
                             int *barrier_ptr,
                             hipStream_t stream,
                             bool all_gather_scale_a)
{
    custom_copy_all_gather_fp8_kernel<<<grid_size, block_size, 0, stream>>>(
        vec_size,
        world_size,
        rank,
        rows,
        cols,
        pp_srcs,
        pp_scales,
        barrier_ptr,
        all_gather_scale_a);
}

void ag_copy_remote_ragged_ring_payload_acquire_impl(
    int grid_size,
    int block_size,
    const void *payload,
    size_t bytes,
    int *internal_signal,
    int *owner_signal,
    int *completion_counter,
    uint32_t signal_value,
    int probe_stride_words,
    int probe_mode,
    hipStream_t stream)
{
    ragged_ring_payload_acquire_kernel<<<grid_size, block_size, 0, stream>>>(
        payload,
        bytes,
        internal_signal,
        owner_signal,
        completion_counter,
        signal_value,
        probe_stride_words,
        probe_mode);
}

void ag_copy_remote_ragged_ring_light_acquire_group_impl(
    int *local_signals,
    uint32_t signal_value,
    int owner_begin,
    int owner_count,
    int rank,
    int body_slot_base,
    int *ready_mask,
    hipStream_t stream)
{
    ragged_ring_light_acquire_group_kernel<<<owner_count, 1, 0, stream>>>(
        local_signals,
        signal_value,
        owner_begin,
        owner_count,
        rank,
        body_slot_base,
        ready_mask);
}

} // namespace hygon_vendor
} // namespace bytedance
