// Pad / crop copy kernel of the backward entry (see flash_train_gdn_bwd_pad.h).  One launch moves up to
// eight tensors; the destination is written completely (data or zero), so the padded workspace
// buffers never hold stale values.
#include "gfx938_device_symbols.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>

#include "flash_train_gdn_bwd_pad.h"

namespace {

constexpr int kThreads = 256;
constexpr long long kMaxBlocks = 1 << 18;

struct PadArgs {
    FlashTrainGDNBwdPadDesc d[FLASH_TRAIN_GDN_BWD_PAD_MAX_DESCS];
};

__global__ void __launch_bounds__(kThreads) gdn_bwd_pad_copy_kernel(PadArgs args) {
    const FlashTrainGDNBwdPadDesc d = args.d[blockIdx.y];
    const size_t total = static_cast<size_t>(d.d[0]) * d.d[1] * d.d[2] * d.d[3];
    for (size_t idx = static_cast<size_t>(blockIdx.x) * kThreads + threadIdx.x; idx < total;
         idx += static_cast<size_t>(gridDim.x) * kThreads) {
        size_t r = idx;
        const int i3 = static_cast<int>(r % d.d[3]); r /= d.d[3];
        const int i2 = static_cast<int>(r % d.d[2]); r /= d.d[2];
        const int i1 = static_cast<int>(r % d.d[1]);
        const int i0 = static_cast<int>(r / d.d[1]);
        // source coordinates of dims 2/3 (exchanged when the copy transposes them)
        const int s2 = d.swap23 ? i3 : i2;
        const int s3 = d.swap23 ? i2 : i3;
        const bool in = i0 < d.a[0] && i1 < d.a[1] && s2 < d.a[2] && s3 < d.a[3];
        const size_t sidx = ((static_cast<size_t>(i0) * d.a[1] + i1) * d.a[2] + s2) * d.a[3] + s3;
        if (d.elem == 2) {
            static_cast<uint16_t*>(d.dst)[idx] = in ? static_cast<const uint16_t*>(d.src)[sidx] : uint16_t(0);
        } else {
            static_cast<float*>(d.dst)[idx] = in ? static_cast<const float*>(d.src)[sidx] : 0.f;
        }
    }
}

}  // namespace

extern "C" int flash_train_gdn_bwd_pad_copy(const FlashTrainGDNBwdPadDesc* descs, int n, void* stream) {
    if (descs == nullptr) return 1;
    if (n <= 0 || n > FLASH_TRAIN_GDN_BWD_PAD_MAX_DESCS) return 2;
    PadArgs args{};
    size_t max_total = 1;
    for (int i = 0; i < n; ++i) {
        const FlashTrainGDNBwdPadDesc& x = descs[i];
        if (x.src == nullptr || x.dst == nullptr) return 1;
        if (x.elem != 2 && x.elem != 4) return 2;
        size_t total = 1;
        for (int j = 0; j < 4; ++j) {
            if (x.a[j] <= 0 || x.d[j] <= 0) return 2;
            total *= static_cast<size_t>(x.d[j]);
        }
        max_total = std::max(max_total, total);
        args.d[i] = x;
    }
    const long long blocks = std::min<long long>((static_cast<long long>(max_total) + kThreads - 1) / kThreads, kMaxBlocks);
    hipLaunchKernelGGL(gdn_bwd_pad_copy_kernel, dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(n)),
                       dim3(kThreads), 0, static_cast<hipStream_t>(stream), args);
    return hipGetLastError() == hipSuccess ? 0 : 5;
}
