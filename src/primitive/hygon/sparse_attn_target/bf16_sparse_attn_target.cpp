// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>

#include <hip/hip_runtime_api.h>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_target/bf16_sparse_attn_target.hpp"

// Device capability and shape applicability belong to host-side configuration.
// Device capability and shape applicability belong to host-side configuration.
namespace ftrain {
namespace {
static inline int dsa_shared_memory_limit();
static inline int sparse_mmac_max_lds();

static inline bool sparse_mmac_shm_fits(int Hq, int D, int topk) {
    const int max_lds = sparse_mmac_max_lds();
    const int Hq_pad  = (Hq + 15) & ~15;
    size_t shm_lds    = (size_t)Hq_pad * (D + 8) * 2 + (16) * (D + 8) * 2 + (size_t)(256) * 8;
    if (Hq <= 64 && shm_lds + 2048 <= (size_t)max_lds) return true;
    const int dk = D / 16;
    size_t shm_g = (size_t)(16) * (D + 8) * 2 + (size_t)(256) * 8;
    return Hq > 0 && Hq <= 128 && shm_g + 2048 <= (size_t)max_lds &&
           (dk == 4 || dk == 8 || dk == 16 || dk == 28 || dk == 32 || dk == 36);
}
}  // namespace

bool bf16SparseAttnTargetSparseMmacShmFits(int query_heads, int head_dim, int selection_width) {
    return sparse_mmac_shm_fits(query_heads, head_dim, selection_width);
}
}  // namespace ftrain

namespace ftrain {
namespace {
static inline int dsa_shared_memory_limit() {
    int device = 0, limit = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess)
        return 0;
    return limit;
}

static inline bool sparse_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx936") != nullptr || strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static inline int sparse_mmac_max_lds() {
    const int max_lds = dsa_shared_memory_limit();
    return max_lds;
}
}  // namespace

bool bf16SparseAttnTargetSparseMmacSupported() { return sparse_mmac_supported(); }

template<class Implementation>
bool bf16SparseAttnTargetAlgorithmFits(int heads, int width) {
    const auto limit = static_cast<std::size_t>(sparse_mmac_max_lds());
    const auto d     = static_cast<std::size_t>(width + 8);
    if ((Implementation::kGlobalQ)) {
        const int dk = width / 16;
        return heads <= 128 && (dk == 4 || dk == 8 || dk == 16 || dk == 28 || dk == 32 || dk == 36) &&
               (16) * d * 2 + (256) * 8 + 2048 <= limit;
    }
    if ((Implementation::kLdsQ)) {
        return heads <= 64 &&
               static_cast<std::size_t>((heads + 15) & ~15) * d * 2 + (16) * d * 2 + (256) * 8 + 2048 <= limit;
    }
    int device = 0;
    hipDeviceProp_t props{};
    return hipGetDevice(&device) == hipSuccess && hipGetDeviceProperties(&props, device) == hipSuccess &&
           (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0);
}

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int HEADS>
Bf16SparseAttnTargetLaunch bindWidth(int width) {
    if constexpr (Implementation::kGlobalQ) {
        switch (width) {
            case 64: return &launchBf16SparseAttnTarget<Implementation, 64, HEADS>;
            case 128: return &launchBf16SparseAttnTarget<Implementation, 128, HEADS>;
            case 256: return &launchBf16SparseAttnTarget<Implementation, 256, HEADS>;
            case 448: return &launchBf16SparseAttnTarget<Implementation, 448, HEADS>;
            case 512: return &launchBf16SparseAttnTarget<Implementation, 512, HEADS>;
            case 576: return &launchBf16SparseAttnTarget<Implementation, 576, HEADS>;
            default: throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured gather width is not instantiated");
        }
    } else {
        return &launchBf16SparseAttnTarget<Implementation, 128, HEADS>;
    }
}

template<class Implementation>
Bf16SparseAttnTargetLaunch bindBf16SparseAttnTarget(int width, int heads) {
    switch (heads) {
        case 8: return bindWidth<Implementation, 8>(width);
        case 16: return bindWidth<Implementation, 16>(width);
        case 32: return bindWidth<Implementation, 32>(width);
        case 64: return bindWidth<Implementation, 64>(width);
        case 128: return bindWidth<Implementation, 128>(width);
        default: return bindWidth<Implementation, 0>(width);
    }
}

bool hasInt32Indexing(const SparseAttnTargetProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKDims(), &problem.getLseDims(), &problem.getTopkIndicesDims(),
          &problem.getTargetDims(), &problem.getSoftmaxScaleDims(), &problem.getTopkIndicesGlobalDims()}) {
        std::uint64_t count = 1;
        for (const std::int64_t dim : *dims) {
            if (dim > std::numeric_limits<std::int32_t>::max()) { return false; }
            if (dim != 0 && count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) /
                                        static_cast<std::uint64_t>(dim)) {
                return false;
            }
            count *= static_cast<std::uint64_t>(dim);
        }
    }
    return true;
}

// The kernel's tile covers at most this many query heads and reads whole
// 16-element head fragments.
constexpr std::uint64_t kMaxQueryHeads    = 128;
constexpr std::uint64_t kHeadDimAlignment = 16;

}  // namespace

template<class Implementation>
const char* Bf16SparseAttnTargetBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16SparseAttnTargetBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16SparseAttnTargetBase<Implementation>::isApplicable(const SparseAttnTargetProblem& problem,
                                                              const Constraints&) const {
    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKAddress())) &
         7) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 8-byte alignment");
    }
    if ((Implementation::kTarget512 || Implementation::kTarget576) &&
        ((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKAddress())) &
         15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "specialized Q/K vectors require 16-byte alignment");
    }

    if (!(bf16SparseAttnTargetAlgorithmFits<Implementation>(static_cast<int>(problem.getQueryHeads()),
                                                            static_cast<int>(problem.getHeadDim())) &&
          (!Implementation::kTarget512 ||
           (problem.getBatch() == 1 && problem.getQueryLength() >= 1024 && problem.getHeadDim() == 512 &&
            (problem.getQueryHeads() == 64 || problem.getQueryHeads() == 128) &&
            (problem.getSelectionWidth() == 512 || problem.getSelectionWidth() == 1024 ||
             problem.getSelectionWidth() == 1152) &&
            problem.getKeyLength() * problem.getHeadDim() * 2 <= 2147483647ULL)) &&
          (!Implementation::kTarget576 ||
           (problem.getBatch() == 1 && problem.getQueryLength() == 4096 && problem.getKeyLength() == 4096 &&
            problem.getHeadDim() == 576 && (problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64) &&
            problem.getSelectionWidth() == 2048)))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getBatch() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTargetNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesGlobalNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTargetIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesGlobalIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTargetAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getTopkIndicesAddress() == nullptr || problem.getTargetAddress() == nullptr ||
        problem.getSoftmaxScaleAddress() == nullptr || problem.getTopkIndicesGlobalAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getBatch() == 0 || problem.getQueryLength() == 0 || problem.getKeyLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty batch, query, or key length has no kernel work");
    }
    if (problem.getQueryHeads() == 0 || problem.getQueryHeads() > kMaxQueryHeads) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the query head count is outside the kernel's tile coverage");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() % kHeadDimAlignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!bf16SparseAttnTargetSparseMmacSupported() ||
        !bf16SparseAttnTargetSparseMmacShmFits(static_cast<int>(problem.getQueryHeads()),
                                               static_cast<int>(problem.getHeadDim()),
                                               static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device or this shape cannot run the sparse MMAC gather path this kernel needs");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16SparseAttnTargetBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Bf16SparseAttnTargetBase<Implementation>::configure(const SparseAttnTargetProblem& problem) {
    q_                   = problem.getQAddress();
    k_                   = problem.getKAddress();
    lse_                 = static_cast<const float*>(problem.getLseAddress());
    topk_indices_        = static_cast<const int*>(problem.getTopkIndicesAddress());
    target_              = static_cast<float*>(problem.getTargetAddress());
    batch_               = static_cast<int>(problem.getBatch());
    query_length_        = static_cast<int>(problem.getQueryLength());
    key_length_          = static_cast<int>(problem.getKeyLength());
    query_heads_         = static_cast<int>(problem.getQueryHeads());
    head_dim_            = static_cast<int>(problem.getHeadDim());
    selection_width_     = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_       = *static_cast<const float*>(problem.getSoftmaxScaleAddress());
    launch_               = bindBf16SparseAttnTarget<Implementation>(head_dim_, query_heads_);
    topk_indices_global_ = *static_cast<const std::uint8_t*>(problem.getTopkIndicesGlobalAddress());
    workspace_bytes_     = 0;
    row_count_offset_    = workspace_bytes_;
    workspace_bytes_ += static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int);
}

template<class Implementation>
void Bf16SparseAttnTargetBase<Implementation>::executeImpl(const Resources& resources) {
    int* const row_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + row_count_offset_);
    bf16SparseAttnTargetZeroWorkspace(
        resources.getStream(), row_count,
        static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int));
    const bool launched =
        launch_(resources.getStream(), q_, k_, lse_, topk_indices_, target_, row_count, batch_, query_length_,
                key_length_, query_heads_, head_dim_, selection_width_, softmax_scale_, topk_indices_global_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnTarget kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16SparseAttnTargetBase<Bf16SparseAttnTarget>;
template class Bf16SparseAttnTargetBase<Bf16SparseAttnTargetGlobalQ>;
template class Bf16SparseAttnTargetBase<Bf16SparseAttnTargetTarget512>;
template class Bf16SparseAttnTargetBase<Bf16SparseAttnTargetTarget576>;
}  // namespace ftrain
