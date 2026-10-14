#pragma once
#include "c10/util/intrusive_ptr.h"
#include <hip/hip_runtime_api.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/torch.h>
#include <vector>
namespace bytedance::hygon_vendor
{

void init_hygon_shm(c10::intrusive_ptr<c10d::ProcessGroup> pg);
torch::Tensor hygon_create_tensor(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg = nullptr);
std::vector<torch::Tensor> hygon_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    c10::intrusive_ptr<c10d::ProcessGroup> pg = nullptr,
    bool malloc_with_flag = false); // 是否申请L2直通显存);
void hygon_barrier_all_on_stream(
    hipStream_t stream,
    c10::optional<std::vector<torch::Tensor>> barrier_tensors = c10::nullopt,
    c10::optional<int> rank = c10::nullopt);

void pyhygon_barrier_all_on_stream(
    intptr_t stream,
    c10::optional<std::vector<torch::Tensor>> barrier_tensors = c10::nullopt,
    c10::optional<int> rank = c10::nullopt);

// suggest use the functions above if possible
std::vector<torch::Tensor> cudaipc_create_tensor_list(
    c10::intrusive_ptr<c10d::ProcessGroup> pg,
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype,
    bool malloc_with_flag = false);

#ifdef FTRAIN_HYGON_SHM_USE_NVSHMEM
std::vector<torch::Tensor> nvshmem_create_tensor_list(
    const std::vector<int64_t> &shape,
    c10::ScalarType dtype);
torch::Tensor nvshmem_create_tensor(const std::vector<int64_t> &shape,
                                    c10::ScalarType dtype);
#endif

} // namespace bytedance::hygon_vendor
