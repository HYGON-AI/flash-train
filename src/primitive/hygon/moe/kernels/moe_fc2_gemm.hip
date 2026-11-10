#include "moe_core.h"
#include "moe_assets.h"
#include "hip/hip_ext.h"
#include "hip_kernels/moe_fc2_gemm.h"
#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace ftrain
{
namespace moe
{
namespace
{

struct __attribute__((packed)) FC2Arguments
{
    const void *input;
    const void *weight;
    const int *gather_index;
    void *output;
    const int *rank_signals;
    const int *tile_info;
    uint32_t problem_count;
    uint32_t n_dim;
    uint32_t k_dim;
    uint32_t topk;
    uint32_t rank;
    uint32_t signal_value;
    uint32_t epilogue_mode;
    uint32_t chunk_size_n;
    uint32_t num_done_groups;
    uint32_t group_major;
    const int *sorted_token_ids;
    int *chunk_done_counts;
    int *chunk_done_flags;
};

static_assert(sizeof(FC2Arguments) == 112);

struct FC2Module
{
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
};

hipFunction_t get_fc2_function(bool weight_nk)
{
    int device = 0;
    CUDA_CHECK(hipGetDevice(&device));
    const char *override_path = std::getenv(
        weight_nk ? "FTRAIN_MOE_FC2_NK_ASM" : "FTRAIN_MOE_FC2_ASM");
    const char *package_dir = std::getenv("FTRAIN_MOE_RS_ASM_DIR_GFX936");
    const std::string kernel_name = weight_nk ? "ftrain_moe_rs_nt" : "ftrain_moe_rs_nn";
    const std::string filename = kernel_name + ".co";
    std::string path;
    if (override_path != nullptr && *override_path != '\0')
    {
        path = override_path;
    }
    else
    {
        if (package_dir != nullptr && *package_dir != '\0')
        {
            path = std::string(package_dir) + "/" + filename;
        }
        else
        {
            path = ::ftrain::moe::assets::resolveCodeObject(filename);
        }
    }
    static std::mutex mutex;
    static std::map<std::pair<int, std::string>, FC2Module> modules;
    std::lock_guard<std::mutex> guard(mutex);
    auto &module = modules[{device, path}];
    if (module.function == nullptr)
    {
        hipDeviceProp_t properties{};
        CUDA_CHECK(hipGetDeviceProperties(&properties, device));
        FTRAIN_CHECK(std::string(properties.gcnArchName).find("gfx936") == 0)
            << "FC2 dedicated ASM currently supports gfx936 only";
        CUDA_CHECK(::ftrain::moe::assets::loadCodeObject(&module.module, path));
        CUDA_CHECK(hipModuleGetFunction(&module.function,
                                        module.module,
                                        kernel_name.c_str()));
    }
    return module.function;
}

__global__ void fc2_pack_input_kernel(const uint4 *__restrict__ input,
                                      uint4 *__restrict__ packed_input,
                                      const int *__restrict__ gather_index,
                                      int sorted_capacity,
                                      int input_rows,
                                      int vectors_per_row)
{
    const int64_t elements =
        static_cast<int64_t>(sorted_capacity) * vectors_per_row;
    for (int64_t index =
             static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<int64_t>(blockDim.x) * gridDim.x)
    {
        const int row = index / vectors_per_row;
        const int column = index % vectors_per_row;
        const int source_row = gather_index[row];
        packed_input[index] =
            source_row >= 0 && source_row < input_rows
                ? input[static_cast<int64_t>(source_row) * vectors_per_row +
                        column]
                : make_uint4(0, 0, 0, 0);
    }
}

__global__ void fc2_build_contract_kernel(const int *__restrict__ tile_info,
                                          int *__restrict__ contract,
                                          int problem_count)
{
    const int problem = blockIdx.x * blockDim.x + threadIdx.x;
    if (problem < problem_count)
    {
        const int *tile = tile_info + problem * 5;
        contract[problem] = tile[0];
        contract[problem_count + problem] = tile[1];
        contract[2 * problem_count + problem] = -1;
        contract[3 * problem_count + problem] = -1;
        contract[4 * problem_count + problem] = tile[2];
    }
}

} // namespace

void moe_fc2_prepare_input(const void *input,
                           void *packed_input,
                           const int *sorted_gather_index,
                           const int *rank_tile_info,
                           int *contract_tile_info,
                           int sorted_capacity,
                           int input_rows,
                           int k_dim,
                           int problem_count,
                           hipStream_t stream)
{
    FTRAIN_CHECK(k_dim > 0 && k_dim % 8 == 0);
    const int64_t vectors = static_cast<int64_t>(sorted_capacity) * (k_dim / 8);
    const int blocks =
        static_cast<int>(std::min<int64_t>((vectors + 255) / 256, 4096));
    hipLaunchKernelGGL(fc2_pack_input_kernel,
                       dim3(blocks),
                       dim3(256),
                       0,
                       stream,
                       static_cast<const uint4 *>(input),
                       static_cast<uint4 *>(packed_input),
                       sorted_gather_index,
                       sorted_capacity,
                       input_rows,
                       k_dim / 8);
    hipLaunchKernelGGL(fc2_build_contract_kernel,
                       dim3((problem_count + 255) / 256),
                       dim3(256),
                       0,
                       stream,
                       rank_tile_info,
                       contract_tile_info,
                       problem_count);
    CUDA_CHECK(hipGetLastError());
}

void moe_fc2_gemm_forward(const void *packed_input,
                          const void *weight,
                          const int *sorted_token_ids,
                          const int *contract_tile_info,
                          void *output,
                          int *chunk_done_counts,
                          int *chunk_done_flags,
                          int problem_count,
                          int output_rows,
                          int n_dim,
                          int k_dim,
                          int chunk_size_n,
                          int num_done_groups,
                          hipStream_t stream,
                          bool weight_nk)
{
    FTRAIN_CHECK(problem_count > 0 && n_dim > 0 && n_dim % 256 == 0);
    FTRAIN_CHECK(k_dim > 0 && k_dim % 16 == 0);
    FTRAIN_CHECK(!weight_nk || k_dim % 64 == 0)
        << "FC2 NK dedicated GEMM requires K divisible by 64";
    FTRAIN_CHECK(chunk_size_n > 0 && chunk_size_n % 256 == 0);
    FC2Arguments arguments{};
    arguments.input = packed_input;
    arguments.weight = weight;
    arguments.output = output;
    arguments.tile_info = contract_tile_info;
    arguments.problem_count = problem_count;
    arguments.n_dim = n_dim;
    arguments.k_dim = k_dim;
    arguments.topk = 1;
    arguments.signal_value = 1;
    arguments.epilogue_mode = 1;
    arguments.chunk_size_n = chunk_size_n;
    arguments.num_done_groups = num_done_groups;
    arguments.group_major = num_done_groups > 0 ? 1 : 0;
    const char *buffer_store = std::getenv("FTRAIN_MOE_FC2_BUFFER_STORE");
    if (static_cast<uint64_t>(output_rows) * n_dim * 2 <= UINT32_MAX &&
        (buffer_store == nullptr || std::string(buffer_store) != "0"))
    {
        arguments.group_major |= 2;
    }
    arguments.sorted_token_ids = sorted_token_ids;
    arguments.chunk_done_counts = chunk_done_counts;
    arguments.chunk_done_flags = chunk_done_flags;
    size_t argument_size = sizeof(arguments);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      &arguments,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argument_size,
                      HIP_LAUNCH_PARAM_END};
    const uint32_t grid_x =
        num_done_groups > 0 ? chunk_size_n / 256 : n_dim / 256;
    const uint32_t grid_y = num_done_groups > 0
                                ? problem_count * num_done_groups
                                : ((problem_count + 7) / 8) * 8;
    const uint32_t block_size = weight_nk ? 768 : 512;
    CUDA_CHECK(hipExtModuleLaunchKernel(get_fc2_function(weight_nk),
                                        grid_x * block_size,
                                        grid_y,
                                        1,
                                        block_size,
                                        1,
                                        1,
                                        0,
                                        stream,
                                        nullptr,
                                        config,
                                        0,
                                        0,
                                        0));
}

} // namespace moe
} // namespace ftrain
