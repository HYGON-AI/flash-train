#include "hygon_vendor.h"
#include "hip_kernels/helper_kernels.h"
#include "hip/hip_runtime.h"

namespace bytedance
{
namespace hygon_vendor
{

struct CudaIpcBarrierAllArgs
{
    int32_t *sync_buffers[MAX_LOCAL_WORLD_SIZE];
    int rank;
    int world_size;
};

__global__ void CudaIpcBarrierAllKernel(CudaIpcBarrierAllArgs args)
{
    int **sync_buffers = args.sync_buffers;
    int world_size = args.world_size;
    int cur_rank = args.rank;
    if (threadIdx.x < world_size)
    {
        // set achieved flag for others
        int *sync_buffer_dst = sync_buffers[threadIdx.x] + cur_rank;
        uint32_t const data = 1;
#pragma unroll 1
        while (atomicCAS_system(sync_buffer_dst, 0, 1) != 0)
        {
        }
        int *wait_ptr = sync_buffers[cur_rank] + threadIdx.x;
#pragma unroll 1
        while (atomicCAS_system(wait_ptr, 1, 0) != 1)
        {
        }
    }
}

void cudaipc_barrier_all_on_stream_impl(hipStream_t stream,
                                        int32_t **sync_buffer_ptr,
                                        int rank,
                                        int world_size)
{
    dim3 grid_dim(1);
    dim3 block_dim(MAX_LOCAL_WORLD_SIZE);
    CudaIpcBarrierAllArgs args;
    args.world_size = world_size;
    args.rank = rank;
    assert(world_size <= MAX_LOCAL_WORLD_SIZE);
    for (int i = 0; i < world_size; i++)
    {
        args.sync_buffers[i] = sync_buffer_ptr[i];
    }
    hipLaunchKernelGGL((CudaIpcBarrierAllKernel),
                       dim3(grid_dim),
                       dim3(block_dim),
                       0,
                       stream,
                       args);
}

} // namespace hygon_vendor
} // namespace bytedance
