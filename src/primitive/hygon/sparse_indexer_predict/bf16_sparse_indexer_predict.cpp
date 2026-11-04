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
#include "flash_train/primitive/hygon/sparse_indexer_predict/bf16_sparse_indexer_predict.hpp"

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

bool bf16SparseIndexerPredictSparseMmacShmFits(int query_heads, int head_dim, int selection_width) {
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

bool bf16SparseIndexerPredictSparseMmacSupported() { return sparse_mmac_supported(); }

template<class Implementation>
bool bf16SparseIndexerPredictAlgorithmFits(int heads, int width) {
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
    return false;
}

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int HEADS>
Bf16SparseIndexerPredictLaunch bindWidth(int width) {
    if constexpr (Implementation::kGlobalQ) {
        switch (width) {
            case 64: return &launchBf16SparseIndexerPredict<Implementation, 64, HEADS>;
            case 128: return &launchBf16SparseIndexerPredict<Implementation, 128, HEADS>;
            case 256: return &launchBf16SparseIndexerPredict<Implementation, 256, HEADS>;
            case 448: return &launchBf16SparseIndexerPredict<Implementation, 448, HEADS>;
            case 512: return &launchBf16SparseIndexerPredict<Implementation, 512, HEADS>;
            case 576: return &launchBf16SparseIndexerPredict<Implementation, 576, HEADS>;
            default: throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured gather width is not instantiated");
        }
    } else {
        return &launchBf16SparseIndexerPredict<Implementation, 128, HEADS>;
    }
}

template<class Implementation>
Bf16SparseIndexerPredictLaunch bindBf16SparseIndexerPredict(int width, int heads) {
    if constexpr (Implementation::kSingle || Implementation::kShared || Implementation::kSharedNormCache) {
        if (width != 128) {
            throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured sparse predict fast path requires D=128");
        }
        switch (heads) {
            case 16: return bindWidth<Implementation, 16>(width);
            case 32: return bindWidth<Implementation, 32>(width);
            case 64: return bindWidth<Implementation, 64>(width);
            default:
                throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                                "configured sparse predict fast path has unsupported heads");
        }
    } else {
        switch (heads) {
            case 8: return bindWidth<Implementation, 8>(width);
            case 16: return bindWidth<Implementation, 16>(width);
            case 32: return bindWidth<Implementation, 32>(width);
            case 64: return bindWidth<Implementation, 64>(width);
            case 128: return bindWidth<Implementation, 128>(width);
            default: return bindWidth<Implementation, 0>(width);
        }
    }
}

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const SparseIndexerPredictProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKDims(), &problem.getWDims(), &problem.getTopkIndicesDims(),
          &problem.getPredictDims(), &problem.getSmScaleDims(), &problem.getTopkIndicesGlobalDims()}) {
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

// The head-dimension bound the indexer predict kernel's tile covers.
constexpr std::uint64_t kMaxHeadDim = 128;

}  // namespace

template<class Implementation>
const char* Bf16SparseIndexerPredictBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16SparseIndexerPredictBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16SparseIndexerPredictBase<Implementation>::isApplicable(const SparseIndexerPredictProblem& problem,
                                                                  const Constraints& constraints) const {
    if constexpr (Implementation::kMmacReduce) {
        if (problem.getHeadDim() > 256 || !isGfx936Or938(constraints.getDeviceId())) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC reduction is limited to small-D gfx936/gfx938 calls");
        }
    }
    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKAddress())) &
         7) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 8-byte alignment");
    }

    if ((Implementation::kSingle || Implementation::kShared || Implementation::kSharedNormCache) &&
        !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires gfx936 or gfx938");
    }

    if ((Implementation::kSingle || Implementation::kShared || Implementation::kSharedNormCache) &&
        ((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getWAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress())) &
         15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires 16-byte aligned vector operands");
    }
    if (!(((problem.getHeadDim() == 128 &&
            (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64) &&
            !problem.getTopkIndicesGlobal()) &&
           (Implementation::kSingle || Implementation::kShared ||
            (Implementation::kSharedNormCache && problem.getQueryHeads() == 32 &&
             problem.getSelectionWidth() == 2048))) ||
          (bf16SparseIndexerPredictAlgorithmFits<Implementation>(static_cast<int>(problem.getQueryHeads()),
                                                                 static_cast<int>(problem.getHeadDim()))))) {
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
        problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getPredictNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesGlobalNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getPredictIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesGlobalIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getPredictAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getWAddress() == nullptr ||
        problem.getTopkIndicesAddress() == nullptr || problem.getPredictAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getTopkIndicesGlobalAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > (Implementation::kMmacReduce ? 256 : kMaxHeadDim) ||
        problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension exceeds the kernel's tile coverage");
    }
    if (problem.getQueryHeads() == 0 || problem.getQueryHeads() > 128) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "query heads must lie in 1..128");
    }
    if (!bf16SparseIndexerPredictSparseMmacSupported() ||
        !bf16SparseIndexerPredictSparseMmacShmFits(static_cast<int>(problem.getQueryHeads()),
                                                   static_cast<int>(problem.getHeadDim()),
                                                   static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16SparseIndexerPredictBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Bf16SparseIndexerPredictBase<Implementation>::configure(const SparseIndexerPredictProblem& problem) {
    q_                   = problem.getQAddress();
    k_                   = problem.getKAddress();
    w_                   = problem.getWAddress();
    topk_indices_        = static_cast<const int*>(problem.getTopkIndicesAddress());
    predict_             = static_cast<float*>(problem.getPredictAddress());
    batch_               = static_cast<int>(problem.getBatch());
    query_length_        = static_cast<int>(problem.getQueryLength());
    key_length_          = static_cast<int>(problem.getKeyLength());
    query_heads_         = static_cast<int>(problem.getQueryHeads());
    head_dim_            = static_cast<int>(problem.getHeadDim());
    selection_width_     = static_cast<int>(problem.getSelectionWidth());
    sm_scale_            = *static_cast<const float*>(problem.getSmScaleAddress());
    launch_               = bindBf16SparseIndexerPredict<Implementation>(head_dim_, query_heads_);
    topk_indices_global_ = *static_cast<const std::uint8_t*>(problem.getTopkIndicesGlobalAddress());
    workspace_bytes_     = 0;
    row_count_offset_    = workspace_bytes_;
    workspace_bytes_ += static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int);
}

template<class Implementation>
void Bf16SparseIndexerPredictBase<Implementation>::executeImpl(const Resources& resources) {
    int* const row_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + row_count_offset_);
    if (Implementation::kLdsQ || Implementation::kGlobalQ)
        bf16SparseIndexerPredictZeroWorkspace(
            resources.getStream(), row_count,
            static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(query_length_) * sizeof(int));
    const bool launched =
        launch_(resources.getStream(), q_, k_, w_, topk_indices_, predict_, row_count, batch_, query_length_,
                key_length_, query_heads_, head_dim_, selection_width_, sm_scale_, topk_indices_global_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseIndexerPredict kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictSingle>;
template class Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictShared>;
template class Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictSharedNormCache>;
template class Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredict>;
template class Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictGlobalQ>;
template class Bf16SparseIndexerPredictBase<Bf16SparseIndexerPredictGlobalQMmacReduce>;
}  // namespace ftrain
