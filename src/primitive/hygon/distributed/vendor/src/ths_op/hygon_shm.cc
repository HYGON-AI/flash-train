
#include "ths_op/hygon_shm.h"
#include "hygon_vendor.h"
#include "hip/hip_runtime.h"
#include "hip_kernels/helper_kernels.h"
#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <c10/hip/HIPFunctions.h>
#include <iostream>
#include <torch/cuda.h>
#include <vector>

#ifdef FTRAIN_HYGON_SHM_USE_NVSHMEM
#include <nvshmemx.h>
#endif

namespace bytedance
{
namespace hygon_vendor
{

#ifdef FTRAIN_HYGON_SHM_USE_NVSHMEM
namespace
{
std::array<const char *, 5> kNvshmemInitStatus = {
    "NVSHMEM_STATUS_NOT_INITIALIZED",
    "NVSHMEM_STATUS_IS_BOOTSTRAPPED",
    "NVSHMEM_STATUS_IS_INITIALIZED",
    "NVSHMEM_STATUS_LIMITED_MPG",
    "NVSHMEM_STATUS_FULL_MPG"};
void check_nvshmem_init()
{
    FTRAIN_HYGON_CHECK(nvshmemx_init_status() >= NVSHMEM_STATUS_IS_INITIALIZED)
        << "nvshmem not initialized: status "
        << kNvshmemInitStatus[nvshmemx_init_status()];
}
} // namespace
torch::Tensor nvshmem_create_tensor(const std::vector<int64_t> &shape,
                                    c10::ScalarType dtype)
{
    check_nvshmem_init();
    auto option_gpu =
        at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(
            c10::hip::current_device());
    auto size =
        torch::elementSize(dtype) *
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    FTRAIN_HYGON_CHECK(size != 0);
    void *ptr = nvshmem_malloc(size);
    FTRAIN_HYGON_CHECK(ptr != nullptr);
    CUDA_CHECK(hipMemset(ptr, 0, size)); // memset the allocated buffer
    return at::from_blob(
        ptr,
        shape,
        [](void *ptr) { nvshmem_free(ptr); },
        option_gpu);
}

std::vector<torch::Tensor> nvshmem_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype)
{
    check_nvshmem_init();
    auto option_gpu =
        at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(
            c10::hip::current_device());
    auto size =
        torch::elementSize(dtype) *
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    FTRAIN_HYGON_CHECK(size != 0);
    int local_world_size = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
    int rank = nvshmem_my_pe();
    int local_rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    std::vector<torch::Tensor> tensors;
    tensors.reserve(local_world_size);
    void *ptr = nvshmem_malloc(size);
    CUDA_CHECK(hipMemset(ptr, 0, size)); // memset the allocated buffer
    FTRAIN_HYGON_CHECK(ptr != nullptr);
    int rank_offset = rank - local_rank;
    for (int i = 0; i < local_world_size; i++)
    {
        // runs this call nvshmem failure, don't know why
        //  nvshmem_team_translate_pe(NVSHMEMX_TEAM_NODE, local_rank, NVSHMEM_TEAM_WORLD)
        int rank_global = i + rank_offset;
        if (rank == rank_global)
        {
            tensors.emplace_back(at::from_blob(
                ptr,
                shape,
                [](void *ptr) { nvshmem_free(ptr); },
                option_gpu));
        }
        else
        {
            void *rptr = nvshmem_ptr(ptr, rank_global);
            FTRAIN_HYGON_CHECK(rptr != nullptr) << "rank " << rank;
            tensors.emplace_back(at::from_blob(rptr, shape, option_gpu));
        }
    }

    return tensors;
}

#endif

std::vector<torch::Tensor> cudaipc_create_tensor_list(
    c10::intrusive_ptr<c10d::ProcessGroup> pg,
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    bool malloc_with_flag)
{
    auto option_gpu =
        at::TensorOptions().dtype(dtype).device(at::kCUDA).device_index(
            c10::hip::current_device());

    FTRAIN_HYGON_CHECK(pg->getSize() <= torch::cuda::device_count())
        << "create_ipc_tensors should only be used intra node";
    auto size =
        torch::elementSize(dtype) *
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    FTRAIN_HYGON_CHECK(size != 0);
    void *ptr = nullptr;
    if (not malloc_with_flag)
    {
        CUDA_CHECK(hipMalloc(&ptr, size));
    }
    else
    {
        CUDA_CHECK(hipExtMallocWithFlags(
            (void **)&ptr,
            size,
            hipDeviceMallocUncached)); // 申请L2直通显存，hipMalloc申请的显存在kernel中会有cache刷新问题
    }
    //TODO hipMemset是否可去掉或overlap
    CUDA_CHECK(hipMemset(ptr, 0, size)); // memset the allocated buffer
    hipIpcMemHandle_t handle;
    CUDA_CHECK(hipIpcGetMemHandle(&handle, ptr));

    auto option_local = at::TensorOptions()
                            .dtype(torch::kUInt8)
                            .device(at::kCUDA)
                            .device_index(c10::hip::current_device());
    auto handle_d = torch::empty({sizeof(hipIpcMemHandle_t)}, option_local);
    CUDA_CHECK(hipMemcpy(handle_d.data_ptr(),
                         &handle,
                         sizeof(hipIpcMemHandle_t),
                         hipMemcpyHostToDevice));
    auto handles_d = torch::empty(
        {static_cast<long>(sizeof(hipIpcMemHandle_t)) * pg->getSize()},
        option_local); //narrowed from type 'unsigned long' to 'long' in initializer
    // ncclKernel_AllGather_Ring_LL_Max__i8
    pg->_allgather_base(handles_d, handle_d)->wait();

    std::vector<hipIpcMemHandle_t> handles_h(pg->getSize());
    CUDA_CHECK(hipMemcpy(handles_h.data(),
                         handles_d.data_ptr(),
                         sizeof(hipIpcMemHandle_t) * pg->getSize(),
                         hipMemcpyDeviceToHost));

    std::vector<void *> ptrs(pg->getSize());
    for (int i = 0; i < pg->getSize(); ++i)
    {
        if (i != pg->getRank())
        {
            CUDA_CHECK(hipIpcOpenMemHandle(&ptrs[i],
                                           handles_h[i],
                                           hipIpcMemLazyEnablePeerAccess));
        }
        else
        {
            ptrs[i] = ptr;
        }
    }

    std::vector<torch::Tensor> tensors;
    for (int i = 0; i < pg->getSize(); ++i)
    {
        torch::Tensor tensor;
        if (i == pg->getRank())
        {
            tensor = at::from_blob(
                ptr,
                shape,
                [](void *ptr) { hipFree(ptr); },
                option_gpu);
        }
        else
        {
            tensor = at::from_blob(
                ptrs[i],
                shape,
                [](void *ptr) { hipIpcCloseMemHandle(ptr); },
                option_gpu);
        }
        tensors.emplace_back(tensor);
    }
    return tensors;
}

void init_hygon_shm(c10::intrusive_ptr<c10d::ProcessGroup> c10_pg)
{
#ifdef FTRAIN_HYGON_SHM_USE_NVSHMEM
    nvshmemx_init_attr_t init_attr;
    init_attr.mpi_comm = (void *)c10_pg.get(); // bad! pretend I'm the MPIComm
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &init_attr);
    int mype = nvshmem_my_pe();
    CHECK(c10_pg->getRank() == mype) << "NVShmem init: rank does not match PE!"
                                     << c10_pg->getRank() << " vs " << mype;
#endif
}

torch::Tensor hygon_create_tensor(const std::vector<int64_t> &shape,
                                 c10::ScalarType dtype,
                                 c10::intrusive_ptr<c10d::ProcessGroup> pg)
{
#ifdef FTRAIN_HYGON_SHM_USE_NVSHMEM
    return nvshmem_create_tensor(shape, dtype);
#else
    FTRAIN_HYGON_CHECK(false && "This line should never be reached");
    return torch::Tensor();
#endif
}
std::vector<torch::Tensor> hygon_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg,
    bool malloc_with_flag)
{
#ifdef FTRAIN_HYGON_SHM_USE_NVSHMEM
    return nvshmem_create_tensor_list(shape, dtype);
#else
    FTRAIN_HYGON_CHECK(pg != nullptr);
    return cudaipc_create_tensor_list(pg, shape, dtype, malloc_with_flag);
#endif
}

void hygon_barrier_all_on_stream(
    hipStream_t stream,
    c10::optional<std::vector<torch::Tensor>> sync_buffers,
    c10::optional<int> rank)
{
#ifdef FTRAIN_HYGON_SHM_USE_NVSHMEM
    nvshmemx_barrier_all_on_stream(stream);
#else
    FTRAIN_HYGON_CHECK(sync_buffers.has_value());
    FTRAIN_HYGON_CHECK(rank.has_value());
    std::vector<int32_t *> sync_buffer_ptrs;
    auto sync_buffers_val = sync_buffers.value();
    FTRAIN_HYGON_CHECK(sync_buffers_val[rank.value()].defined());
    int world_size = sync_buffers_val.size();
    for (int i = 0; i < sync_buffers_val.size(); i++)
    {
        sync_buffer_ptrs.push_back(
            reinterpret_cast<int32_t *>(sync_buffers_val[i].data_ptr()));
    }
    cudaipc_barrier_all_on_stream_impl(stream,
                                       sync_buffer_ptrs.data(),
                                       rank.value(),
                                       world_size);
#endif
}

void pyhygon_barrier_all_on_stream(
    intptr_t stream,
    c10::optional<std::vector<torch::Tensor>> sync_buffers,
    c10::optional<int> rank)
{
    hygon_barrier_all_on_stream((hipStream_t)stream, sync_buffers, rank);
}

} // namespace hygon_vendor
} // namespace bytedance
