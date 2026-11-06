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
#include "flash_train/primitive/hygon/dense_indexer_score/mixed_fp32_dense_indexer_score.hpp"

// Device capability and shape applicability belong to host-side configuration.
// Select the storage-conversion launch on the host.
namespace ftrain {
void castMixedFp32DenseIndexerScore(const void* source, void* destination, FTrainNumericType source_type,
                                    FTrainNumericType destination_type, std::uint64_t elements, FTrainStream stream) {
    if (elements == 0) { return; }
    if (source_type == FTRAIN_NUMERIC_TYPE_BF16 && destination_type == FTRAIN_NUMERIC_TYPE_FP32) {
        launchMixedFp32DenseIndexerScoreCastBf16ToFp32(source, destination, elements, stream);
    } else if (source_type == FTRAIN_NUMERIC_TYPE_FP16 && destination_type == FTRAIN_NUMERIC_TYPE_FP32) {
        launchMixedFp32DenseIndexerScoreCastFp16ToFp32(source, destination, elements, stream);
    } else if (source_type == FTRAIN_NUMERIC_TYPE_FP32 && destination_type == FTRAIN_NUMERIC_TYPE_BF16) {
        launchMixedFp32DenseIndexerScoreCastFp32ToBf16(source, destination, elements, stream);
    } else if (source_type == FTRAIN_NUMERIC_TYPE_FP32 && destination_type == FTRAIN_NUMERIC_TYPE_FP16) {
        launchMixedFp32DenseIndexerScoreCastFp32ToFp16(source, destination, elements, stream);
    } else {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "unsupported mixed-storage conversion");
    }
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

static inline bool dsa_shared_memory_fits(size_t bytes) {
    return bytes <= static_cast<size_t>(dsa_shared_memory_limit());
}

static inline bool dense_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx936") != nullptr || strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static size_t dense_mmac_shm(int Hq, int D, int Sk, bool bwd) {
    int Hq_pad  = (Hq + 15) & ~15;
    int dd      = bwd ? D : (D < (128) ? D : (128));
    size_t sz   = (size_t)(32) * (dd + 8) * 2;  // k_lds
    sz         += (size_t)Sk * 4;               // score
    if (bwd) {
        // K, score-gradient, reduction and BF16 dS tiles; Q stays in registers.
        sz += (256) * 4;                        // red
        sz += (size_t)Hq_pad * ((32) + 2) * 2;  // a_lds (ALDS = 34)
    } else {
        sz += (size_t)Hq_pad * (dd + 8) * 2;  // q_lds (forward kernels)
        sz += (256) * 4;                      // red (denom)
    }
    return sz;
}

static inline bool dense_mmac_shm_fits(int Hq, int D, int Sk, bool bwd) {
    const int max_lds = dsa_shared_memory_limit();
    if (!bwd) return dense_mmac_shm(Hq, D, Sk, false) <= (size_t)max_lds;
    int Hq_pad = (Hq + 15) & ~15;
    return dense_mmac_shm(Hq, D, Sk, true) <= (size_t)max_lds;
}

template<typename C>
static inline size_t dense_mmac_dtype_shm(int score_stride) {
    constexpr int dchunk = (sizeof(C) == 4 ? 64 : 128);
    constexpr int pad    = sizeof(C) == 4 ? 10 : 2;
    constexpr int ld     = dchunk + pad;
    return (size_t)(64 + (32)) * ld * sizeof(C) + (size_t)score_stride * sizeof(float) + (size_t)(256) * sizeof(float);
}

static inline bool dense_mmac_storage_fits(int heads, int width, int keys, bool packed) {
    if (heads <= 0 || heads > 128 || width <= 0 || width > 576 || width % 16 || keys <= 0) return false;
    if (!dense_mmac_shm_fits(heads, width, keys, false)) return false;
    { return dsa_shared_memory_fits(dense_mmac_dtype_shm<float>(keys)); }
}
}  // namespace

bool mixedFp32DenseIndexerScoreLaunchFits(int heads, int width, int keys) {
    return dense_mmac_storage_fits(heads, width, keys, false);
}

bool mixedFp32DenseIndexerScoreDenseMmacSupported() { return dense_mmac_supported(); }

namespace {

bool hasInt32Indexing(const DenseIndexerScoreProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getQDims(), &problem.getKDims(), &problem.getWDims(), &problem.getOutDims(),
                             &problem.getDenomDims(), &problem.getRatioDims(), &problem.getSmScaleDims()}) {
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

// The MMAC score tile covers head dimensions that are multiples of 16 up to
// the largest Q/K width the fused sparse-attention kernels stage.
constexpr std::uint64_t kMaxHeadDim       = 576;
constexpr std::uint64_t kHeadDimAlignment = 16;

}  // namespace

const char* MixedFp32DenseIndexerScore::getName() const noexcept { return "MixedFp32DenseIndexerScore"; }

std::unique_ptr<PrimitiveBase> MixedFp32DenseIndexerScore::clone() const {
    return std::make_unique<MixedFp32DenseIndexerScore>(*this);
}

Result MixedFp32DenseIndexerScore::isApplicable(const DenseIndexerScoreProblem& problem, const Constraints&) const {
    if (problem.getBatch() == 0 || problem.getQueryHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getKNumericType() == problem.getQNumericType() &&
        problem.getWNumericType() == problem.getQNumericType()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires mixed input storage types");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if ((problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        (problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        (problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_FP32 &&
         problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_FP16 &&
         problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_BF16) ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getDenomNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDenomIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) %
                (problem.getQNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) %
                (problem.getKNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWAddress()) %
                (problem.getWNumericType() == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t)) !=
            0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDenomAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // Check required scalar addresses here as well as in StorageView validation.
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getWAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getDenomAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getRatioAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.hasSingleKvHead()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the kernel supports one KV head"); }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % kHeadDimAlignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!mixedFp32DenseIndexerScoreDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!mixedFp32DenseIndexerScoreLaunchFits(static_cast<int>(problem.getQueryHeads()),
                                              static_cast<int>(problem.getHeadDim()),
                                              static_cast<int>(problem.getKeyLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t MixedFp32DenseIndexerScore::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void MixedFp32DenseIndexerScore::configure(const DenseIndexerScoreProblem& problem) {
    q_            = problem.getQAddress();
    k_            = problem.getKAddress();
    w_            = problem.getWAddress();
    out_          = static_cast<float*>(problem.getOutAddress());
    denom_        = static_cast<float*>(problem.getDenomAddress());
    batch_        = static_cast<int>(problem.getBatch());
    query_length_ = static_cast<int>(problem.getQueryLength());
    key_length_   = static_cast<int>(problem.getKeyLength());
    query_heads_  = static_cast<int>(problem.getQueryHeads());
    head_dim_     = static_cast<int>(problem.getHeadDim());
    ratio_        = static_cast<int>(problem.getRatio());
    sm_scale_     = *static_cast<const float*>(problem.getSmScaleAddress());

    // Conversion buffers follow this implementation's FP32 kernel scratch.
    workspace_bytes_ = 0;
    q_type_          = problem.getQNumericType();
    q_elements_      = 1;
    for (const auto dim : problem.getQDims()) { q_elements_ *= static_cast<std::uint64_t>(dim); }
    q_conversion_offset_ = workspace_bytes_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += q_elements_ * sizeof(float); }
    k_type_     = problem.getKNumericType();
    k_elements_ = 1;
    for (const auto dim : problem.getKDims()) { k_elements_ *= static_cast<std::uint64_t>(dim); }
    k_conversion_offset_ = workspace_bytes_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += k_elements_ * sizeof(float); }
    w_type_     = problem.getWNumericType();
    w_elements_ = 1;
    for (const auto dim : problem.getWDims()) { w_elements_ *= static_cast<std::uint64_t>(dim); }
    w_conversion_offset_ = workspace_bytes_;
    if (w_type_ != FTRAIN_NUMERIC_TYPE_FP32) { workspace_bytes_ += w_elements_ * sizeof(float); }
}

void MixedFp32DenseIndexerScore::executeImpl(const Resources& resources) {
    const void* q_compute = q_;
    if (q_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + q_conversion_offset_;
        castMixedFp32DenseIndexerScore(q_, converted, q_type_, FTRAIN_NUMERIC_TYPE_FP32, q_elements_,
                                       resources.getStream());
        q_compute = converted;
    }
    const void* k_compute = k_;
    if (k_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + k_conversion_offset_;
        castMixedFp32DenseIndexerScore(k_, converted, k_type_, FTRAIN_NUMERIC_TYPE_FP32, k_elements_,
                                       resources.getStream());
        k_compute = converted;
    }
    const void* w_compute = w_;
    if (w_type_ != FTRAIN_NUMERIC_TYPE_FP32) {
        void* converted = static_cast<unsigned char*>(resources.getWorkspace()) + w_conversion_offset_;
        castMixedFp32DenseIndexerScore(w_, converted, w_type_, FTRAIN_NUMERIC_TYPE_FP32, w_elements_,
                                       resources.getStream());
        w_compute = converted;
    }

    const bool launched = launchMixedFp32DenseIndexerScore(resources.getStream(), q_compute, k_compute, w_compute, out_,
                                                           denom_, batch_, query_length_, key_length_, query_heads_,
                                                           /*kv_heads=*/1, head_dim_, ratio_, sm_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "DenseIndexerScore kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
