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
#include "flash_train/primitive/hygon/indexer_fwd/fp16_indexer_fwd.hpp"

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

    { return dsa_shared_memory_fits(dense_mmac_dtype_shm<std::uint16_t>(keys)); }
}
}  // namespace

bool fp16IndexerFwdLaunchFits(int heads, int width, int keys) {
    return dense_mmac_storage_fits(heads, width, keys, false);
}

bool fp16IndexerFwdDenseMmacSupported() { return dense_mmac_supported(); }

namespace {

bool hasInt32Indexing(const IndexerFwdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getQDims(), &problem.getKDims(), &problem.getWDims(), &problem.getScoresDims(),
                             &problem.getRatioDims(), &problem.getSmScaleDims()}) {
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

// The kernel's score tile is MQA and reads whole 16-element head fragments.
constexpr std::uint64_t kMaxHeadDim      = 512;
constexpr std::uint64_t kHeadDimMultiple = 16;

}  // namespace

const char* Fp16IndexerFwd::getName() const noexcept { return "Fp16IndexerFwd"; }

std::unique_ptr<PrimitiveBase> Fp16IndexerFwd::clone() const { return std::make_unique<Fp16IndexerFwd>(*this); }

// The support checklist, one IndexerFwdProblem getter group at a time:
// compute and element types, layouts, addresses, and shape. Memory placement
// and the port-shape relations are constructor-guaranteed.
Result Fp16IndexerFwd::isApplicable(const IndexerFwdProblem& problem, const Constraints&) const {
    // dsa_copy_as stages complete vectors, not just individual elements.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKAddress())) & 15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 16-byte alignment");
    }

    if (problem.getBatch() == 0 || problem.getQueryHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getScoresNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getScoresIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getScoresAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getWAddress() == nullptr ||
        problem.getScoresAddress() == nullptr || problem.getSmScaleAddress() == nullptr ||
        problem.getRatioAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getKeyLength() == 0 || problem.getQueryLength() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % kHeadDimMultiple != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!fp16IndexerFwdDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!fp16IndexerFwdLaunchFits(static_cast<int>(problem.getQueryHeads()), static_cast<int>(problem.getHeadDim()),
                                  static_cast<int>(problem.getKeyLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Fp16IndexerFwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp16IndexerFwd::configure(const IndexerFwdProblem& problem) {
    q_            = problem.getQAddress();
    k_            = problem.getKAddress();
    w_            = problem.getWAddress();
    scores_       = static_cast<float*>(problem.getScoresAddress());
    batch_        = static_cast<int>(problem.getBatch());
    query_length_ = static_cast<int>(problem.getQueryLength());
    key_length_   = static_cast<int>(problem.getKeyLength());
    query_heads_  = static_cast<int>(problem.getQueryHeads());
    head_dim_     = static_cast<int>(problem.getHeadDim());
    ratio_        = static_cast<int>(problem.getRatio());
    sm_scale_     = *static_cast<const float*>(problem.getSmScaleAddress());
}

void Fp16IndexerFwd::executeImpl(const Resources& resources) {
    const bool launched = launchFp16IndexerFwd(resources.getStream(), q_, k_, w_, scores_, batch_, query_length_,
                                               key_length_, query_heads_, /*kv_heads=*/1, head_dim_, ratio_, sm_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerFwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
