// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <hip/hip_runtime.h>


namespace ftrain {
namespace {

__global__ void transposeState(const float* source, float* target, std::size_t total,
                               int k, int v, bool to_v_first) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const std::size_t head = index / (static_cast<std::size_t>(k) * v);
    const int key = (index / v) % k, value = index % v;
    const std::size_t swapped = (head * v + value) * k + key;
    if (to_v_first) target[swapped] = source[index];
    else target[index] = source[swapped];
}

}  // namespace

int transposeGdnState(const float* source, float* target, int b, int hv, int k, int v,
                      bool to_v_first, hipStream_t stream) {
    const std::size_t total = static_cast<std::size_t>(b) * hv * k * v;
    hipLaunchKernelGGL(transposeState, dim3((total + 255) / 256), dim3(256), 0, stream,
                       source, target, total, k, v, to_v_first);
    return hipGetLastError() == hipSuccess ? 0 : 5;
}

}  // namespace ftrain
