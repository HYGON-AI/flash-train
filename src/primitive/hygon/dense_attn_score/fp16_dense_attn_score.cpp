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
#include "flash_train/primitive/hygon/dense_attn_score/fp16_dense_attn_score.hpp"

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

bool fp16DenseAttnScoreLaunchFits(int heads, int width, int keys) {
    return dense_mmac_storage_fits(heads, width, keys, false);
}

bool fp16DenseAttnScoreDenseMmacSupported() { return dense_mmac_supported(); }

namespace {

bool hasInt32Indexing(const DenseAttnScoreProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getQDims(), &problem.getKDims(), &problem.getLseDims(), &problem.getOutDims(),
                             &problem.getDenomDims(), &problem.getRatioDims(), &problem.getSoftmaxScaleDims()}) {
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

constexpr std::uint64_t kMaxHeadDim       = 576;
constexpr std::uint64_t kHeadDimAlignment = 16;

}  // namespace

const char* Fp16DenseAttnScore::getName() const noexcept { return "Fp16DenseAttnScore"; }

std::unique_ptr<PrimitiveBase> Fp16DenseAttnScore::clone() const { return std::make_unique<Fp16DenseAttnScore>(*this); }

Result Fp16DenseAttnScore::isApplicable(const DenseAttnScoreProblem& problem, const Constraints&) const {
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
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getDenomNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDenomIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDenomAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // StorageView rejects null rank-zero scalars before Problem construction.
    // DenseAttnScoreProblem requires softmax_scale to be a host scalar, so it is non-null here.
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getDenomAddress() == nullptr ||
        problem.getSoftmaxScaleAddress() == nullptr || problem.getRatioAddress() == nullptr) {
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
    if (!fp16DenseAttnScoreDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!fp16DenseAttnScoreLaunchFits(static_cast<int>(problem.getQueryHeads()), static_cast<int>(problem.getHeadDim()),
                                      static_cast<int>(problem.getKeyLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Fp16DenseAttnScore::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp16DenseAttnScore::configure(const DenseAttnScoreProblem& problem) {
    q_             = problem.getQAddress();
    k_             = problem.getKAddress();
    lse_           = static_cast<const float*>(problem.getLseAddress());
    out_           = static_cast<float*>(problem.getOutAddress());
    denom_         = static_cast<float*>(problem.getDenomAddress());
    batch_         = static_cast<int>(problem.getBatch());
    query_length_  = static_cast<int>(problem.getQueryLength());
    key_length_    = static_cast<int>(problem.getKeyLength());
    query_heads_   = static_cast<int>(problem.getQueryHeads());
    head_dim_      = static_cast<int>(problem.getHeadDim());
    ratio_         = static_cast<int>(problem.getRatio());
    softmax_scale_ = *static_cast<const float*>(problem.getSoftmaxScaleAddress());
}

void Fp16DenseAttnScore::executeImpl(const Resources& resources) {
    const bool launched =
        launchFp16DenseAttnScore(resources.getStream(), q_, k_, lse_, out_, denom_, batch_, query_length_, key_length_,
                                 query_heads_, /*kv_heads=*/1, head_dim_, ratio_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "DenseAttnScore kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
