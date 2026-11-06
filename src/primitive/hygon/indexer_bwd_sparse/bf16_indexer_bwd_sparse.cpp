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
#include "flash_train/primitive/hygon/indexer_bwd_sparse/bf16_indexer_bwd_sparse.hpp"

// Device capability and shape applicability belong to host-side configuration.
namespace ftrain {
namespace {
static inline int dsa_shared_memory_limit() {
    int device = 0, limit = 0;
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&limit, hipDeviceAttributeMaxSharedMemoryPerBlock, device) != hipSuccess)
        return 0;
    return limit;
}

static inline bool dsa_shared_memory_fits(size_t bytes) {
    return bytes <= static_cast<size_t>(dsa_shared_memory_limit());
}

static inline bool dense_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx936") != nullptr || strstr(prop.gcnArchName, "gfx938") != nullptr;
}
}  // namespace

template<class Implementation>
bool bf16IndexerBwdSparseAlgorithmFits(int heads, int width, int columns) {
    if (columns <= 0 || columns > 2048) { return false; }
    std::size_t bytes =
        16ULL * (width + 8) * 2 + static_cast<std::size_t>(columns) * 8 + 256 * 4 + 4 * 320 * 2 + heads * 4;
    if ((Implementation::kCachedScores || Implementation::kCachedScoresCast) ||
        (Implementation::kTwoWave || Implementation::kTwoWaveCast)) {
        bytes -= columns * 8;
    }
    return dsa_shared_memory_fits(bytes);
}

bool bf16IndexerBwdSparseDenseMmacSupported() { return dense_mmac_supported(); }

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int NW, bool DEVICE_SCALE>
Bf16IndexerBwdSparseLaunch bindWidth(int width) {
    if constexpr (Implementation::kGeneric || Implementation::kGenericCast) {
        switch (width) {
            case 16: return &launchBf16IndexerBwdSparse<Implementation, 16, NW, DEVICE_SCALE>;
            case 32: return &launchBf16IndexerBwdSparse<Implementation, 32, NW, DEVICE_SCALE>;
            case 48: return &launchBf16IndexerBwdSparse<Implementation, 48, NW, DEVICE_SCALE>;
            case 64: return &launchBf16IndexerBwdSparse<Implementation, 64, NW, DEVICE_SCALE>;
            case 80: return &launchBf16IndexerBwdSparse<Implementation, 80, NW, DEVICE_SCALE>;
            case 96: return &launchBf16IndexerBwdSparse<Implementation, 96, NW, DEVICE_SCALE>;
            case 112: return &launchBf16IndexerBwdSparse<Implementation, 112, NW, DEVICE_SCALE>;
            case 128: return &launchBf16IndexerBwdSparse<Implementation, 128, NW, DEVICE_SCALE>;
            default:
                throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured sparse indexer width is not instantiated");
        }
    } else {
        return &launchBf16IndexerBwdSparse<Implementation, 128, NW, DEVICE_SCALE>;
    }
}

template<class Implementation, bool DEVICE_SCALE>
Bf16IndexerBwdSparseLaunch bindHeads(int width, int heads) {
    switch ((heads + 15) / 16) {
        case 1: return bindWidth<Implementation, 1, DEVICE_SCALE>(width);
        case 2: return bindWidth<Implementation, 2, DEVICE_SCALE>(width);
        case 3: return bindWidth<Implementation, 3, DEVICE_SCALE>(width);
        case 4: return bindWidth<Implementation, 4, DEVICE_SCALE>(width);
        default: throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured sparse indexer heads are not instantiated");
    }
}

template<class Implementation>
Bf16IndexerBwdSparseLaunch bindBf16IndexerBwdSparse(int width, int heads, bool device_scale) {
    return device_scale ? bindHeads<Implementation, true>(width, heads)
                        : bindHeads<Implementation, false>(width, heads);
}

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const IndexerBwdSparseProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getIndexQDims(), &problem.getWeightsDims(), &problem.getIndexKDims(), &problem.getAttnScoreDims(),
          &problem.getIndexScoreDims(), &problem.getTopkIndicesDims(), &problem.getGradAttnScoreDims(),
          &problem.getDqDims(), &problem.getDwDims(), &problem.getDkDims(), &problem.getSmScaleDims(),
          &problem.getLossCoeffDims(), &problem.getGradLossDims(), &problem.getTopkIndicesGlobalDims()}) {
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

// One completion counter guards the dk cast.
constexpr std::uint64_t kCounterCount = 1;

}  // namespace

template<class Implementation>
const char* Bf16IndexerBwdSparseBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16IndexerBwdSparseBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16IndexerBwdSparseBase<Implementation>::isApplicable(const IndexerBwdSparseProblem& problem,
                                                              const Constraints& constraints) const {
    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDkAddress())) &
         7) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 8-byte alignment");
    }
    if (!(Implementation::kGeneric || Implementation::kGenericCast) &&
        (reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) & 15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "prefetched K vectors require 16-byte alignment");
    }

    if ((Implementation::kPaddedBufferCast || Implementation::kPaddedFlatCast) &&
        ((reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getWeightsAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDwAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDkAddress())) &
         15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "padded bundle requires aligned vector operands");
    }
    if ((!(Implementation::kGeneric || Implementation::kGenericCast)) && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires gfx936 or gfx938");
    }

    const bool alignedShape =
        problem.getHeadDim() == 128 && problem.getHeads() == 64 && problem.getSelectionWidth() % 16 == 0;
    const bool cachedShape = problem.getHeadDim() == 128 && problem.getHeads() == 32 &&
                             problem.getSelectionWidth() == 2048 && problem.getQueryLength() == 4096 &&
                             problem.getKeyLength() == 4096;
    const bool twoWaveShape =
        problem.getHeadDim() == 128 && problem.getHeads() == 64 &&
        ((problem.getSelectionWidth() == 1024 && problem.getQueryLength() >= 2048) ||
         (problem.getSelectionWidth() == 2048 && problem.getQueryLength() == 4096 && problem.getKeyLength() == 4096));
    const bool paddedShape = problem.getHeadDim() == 128 && problem.getHeads() == 64 &&
                             problem.getSelectionWidth() == 2048 && problem.getQueryLength() == 4096 &&
                             problem.getKeyLength() == 4096 && !problem.getTopkIndicesGlobal();
    const bool paddedFits =
        paddedShape && (Implementation::kPaddedFlatCast ||
                        (Implementation::kPaddedBufferCast &&
                         problem.getBatch() * problem.getKeyLength() * 128 * sizeof(float) <= 2147483647ULL));
    if (!((Implementation::kGeneric || Implementation::kGenericCast) ||
          ((Implementation::kAligned || Implementation::kAlignedCast) && alignedShape) ||
          ((Implementation::kCachedScores || Implementation::kCachedScoresCast) && cachedShape) ||
          ((Implementation::kTwoWave || Implementation::kTwoWaveCast) && twoWaveShape) || paddedFits)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getBatch() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    // Validate type, layout and alignment even when the output aliases its input.
    if (problem.getGradAttnScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradAttnScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "in-place score gradient requires continuous FP32 storage");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getIndexQNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getWeightsNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getIndexKNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getAttnScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getIndexScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getDqNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDwNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDkNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getLossCoeffNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradLossNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesGlobalNumericType() != FTRAIN_NUMERIC_TYPE_UINT8) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getIndexQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDqIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDwIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLossCoeffIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradLossIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesGlobalIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightsAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradAttnScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDwAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDkAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLossCoeffAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradLossAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // Check required scalar addresses here as well as in StorageView validation.
    if (problem.getIndexQAddress() == nullptr || problem.getWeightsAddress() == nullptr ||
        problem.getIndexKAddress() == nullptr || problem.getAttnScoreAddress() == nullptr ||
        problem.getIndexScoreAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getDqAddress() == nullptr || problem.getDwAddress() == nullptr || problem.getDkAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getLossCoeffAddress() == nullptr ||
        problem.getGradLossAddress() == nullptr || problem.getGradAttnScoreAddress() == nullptr ||
        problem.getTopkIndicesGlobalAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.isInPlace()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "grad_attn_score must alias attn_score"); }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > 128 || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (problem.getHeads() == 0 || problem.getHeads() > 64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the head count is outside the kernel's tile coverage");
    }
    if (!bf16IndexerBwdSparseDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!bf16IndexerBwdSparseAlgorithmFits<Implementation>(static_cast<int>(problem.getHeads()),
                                                           static_cast<int>(problem.getHeadDim()),
                                                           static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16IndexerBwdSparseBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Bf16IndexerBwdSparseBase<Implementation>::configure(const IndexerBwdSparseProblem& problem) {
    index_q_      = problem.getIndexQAddress();
    weights_      = problem.getWeightsAddress();
    index_k_      = problem.getIndexKAddress();
    index_score_  = problem.getIndexScoreAddress();
    topk_indices_ = static_cast<const int*>(problem.getTopkIndicesAddress());
    // The kernel reads the target and writes its gradient through one buffer,
    // which the Problem guarantees to be the same address.
    score_gradient_  = static_cast<float*>(problem.getGradAttnScoreAddress());
    dq_              = problem.getDqAddress();
    dw_              = problem.getDwAddress();
    dk_              = problem.getDkAddress();
    batch_           = static_cast<int>(problem.getBatch());
    query_length_    = static_cast<int>(problem.getQueryLength());
    key_length_      = static_cast<int>(problem.getKeyLength());
    heads_           = static_cast<int>(problem.getHeads());
    head_dim_        = static_cast<int>(problem.getHeadDim());
    selection_width_ = static_cast<int>(problem.getSelectionWidth());
    sm_scale_        = *static_cast<const float*>(problem.getSmScaleAddress());
    loss_coeff_      = *static_cast<const float*>(problem.getLossCoeffAddress());
    grad_loss_device_ =
        problem.isGradLossHostMemory() ? nullptr : static_cast<const float*>(problem.getGradLossAddress());
    launch_              = bindBf16IndexerBwdSparse<Implementation>(head_dim_, heads_, grad_loss_device_ != nullptr);
    grad_loss_ = problem.isGradLossHostMemory() ? *static_cast<const float*>(problem.getGradLossAddress()) : 1.0F;
    topk_indices_global_ = *static_cast<const std::uint8_t*>(problem.getTopkIndicesGlobalAddress());

    workspace_bytes_    = 0;
    dk_float_offset_    = workspace_bytes_;
    workspace_bytes_   += static_cast<std::uint64_t>(problem.getBatch()) * static_cast<std::uint64_t>(key_length_) *
                          static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += ((Implementation::kPaddedBufferCast || Implementation::kPaddedFlatCast)
                               ? static_cast<std::uint64_t>(batch_) * query_length_ + 2
                               : kCounterCount) *
                          sizeof(int);
}

template<class Implementation>
void Bf16IndexerBwdSparseBase<Implementation>::executeImpl(const Resources& resources) {
    float* const dk_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dk_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    const std::uint64_t dk_float_bytes = static_cast<std::uint64_t>(batch_) * static_cast<std::uint64_t>(key_length_) *
                                         static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    bf16IndexerBwdSparseZeroWorkspace(resources.getStream(), dk_float, dk_float_bytes);
    bf16IndexerBwdSparseZeroWorkspace(
        resources.getStream(), done_count,
        ((Implementation::kPaddedBufferCast || Implementation::kPaddedFlatCast) ? 2 : kCounterCount) * sizeof(int));

    const bool launched = launch_(resources.getStream(), index_q_, weights_, index_k_, score_gradient_,
                                  static_cast<const float*>(index_score_), topk_indices_, dq_, dw_, dk_, dk_float,
                                  done_count, batch_, query_length_, key_length_, heads_, head_dim_, selection_width_,
                                  sm_scale_, loss_coeff_, grad_loss_, topk_indices_global_, grad_loss_device_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerBwdSparse kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparsePaddedBufferCast>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparsePaddedFlatCast>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparse>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseGenericCast>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseAligned>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseAlignedCast>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseCachedScores>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseCachedScoresCast>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseTwoWave>;
template class Bf16IndexerBwdSparseBase<Bf16IndexerBwdSparseTwoWaveCast>;
}  // namespace ftrain
