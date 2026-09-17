#include <algorithm>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "flash_train/error.hpp"
#include "flash_train/ops/gemm/problem.hpp"

namespace ftrain {
namespace {

constexpr std::uint32_t kThreadsPerBlock = 256;
constexpr std::uint32_t kMaxBlockCount   = 4096;

__global__ void fp32GemmKernel(const float* a, const float* b, const float* c, const float* alpha, const float* beta,
                               float* d, std::uint64_t m, std::uint64_t n, std::uint64_t k) {
    const std::uint64_t thread_index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t thread_count = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    const std::uint64_t num_outputs  = m * n;

    for (std::uint64_t index = thread_index; index < num_outputs; index += thread_count) {
        const std::uint64_t row    = index / n;
        const std::uint64_t column = index % n;
        float accumulator          = 0.0F;
        for (std::uint64_t depth = 0; depth < k; ++depth) { accumulator += a[row * k + depth] * b[depth * n + column]; }
        d[index] = *alpha * accumulator + *beta * c[index];
    }
}

}  // namespace

void launchFp32Gemm(const float* a, const float* b, const float* c, const float* alpha, const float* beta, float* d,
                    std::uint64_t m, std::uint64_t n, std::uint64_t k, FTrainStream stream) {
    const std::uint64_t num_outputs = m * n;
    if (num_outputs == 0) { return; }

    const std::uint64_t required_block_count = (num_outputs - 1) / kThreadsPerBlock + 1;
    const auto block_count =
        static_cast<std::uint32_t>(std::min(required_block_count, static_cast<std::uint64_t>(kMaxBlockCount)));

    hipLaunchKernelGGL(fp32GemmKernel, dim3(block_count), dim3(kThreadsPerBlock), 0, stream, a, b, c, alpha, beta, d, m,
                       n, k);

    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "FP32 GEMM kernel launch failed: %s", hipGetErrorString(status));
    }
}

}  // namespace ftrain
