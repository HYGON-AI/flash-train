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
#include "flash_train/primitive/hygon/dense_attn_score/bf16_dense_attn_score.hpp"

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

static size_t dense_mmac_shm_largeD_tpl(int Hq, int D, int Sk) {
    int Hq_pad  = (Hq + 15) & ~15;
    int dd      = (D < (128) ? D : (128)) + 8;
    int qk_rows = (Hq_pad > 2 * (32)) ? Hq_pad : 2 * (32);
    return (size_t)Sk * 4 + (size_t)(256) * 4 + (size_t)qk_rows * dd * 2;
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

static inline size_t dense_mmac_shm_v2(int D, int Sk) {
    return (size_t)(32) * (D + 8) * 2 + (size_t)Sk * 4 + (256) * 4;
}
}  // namespace

template<class Implementation>
bool bf16DenseAttnScoreLaunchFits(int heads, int width, int keys) {
    if (heads <= 0 || heads > 128 || width <= 0 || width > 576 || width % 16 || keys <= 0) return false;
    if (width > 128 && width != 256 && width != 448 && width != 512 && width != 576) return false;
    // Fixed-tile algorithms do not allocate a score row in LDS. Check the
    // selected implementation's footprint, not the historical launch heuristic.
    if constexpr (Implementation::kRingTeacher) {
        return dsa_shared_memory_fits(4 * 2048 * sizeof(std::uint16_t));
    } else if constexpr (Implementation::kLargeD) {
        return dsa_shared_memory_fits(dense_mmac_shm_largeD_tpl(heads, width, keys));
    } else if constexpr (Implementation::kStaged) {
        return dsa_shared_memory_fits(dense_mmac_shm(heads, width, keys, false));
    } else if constexpr (Implementation::kHoisted) {
        return dsa_shared_memory_fits(dense_mmac_shm_v2(width, keys));
    }
    return false;
}

bool bf16DenseAttnScoreDenseMmacSupported() { return dense_mmac_supported(); }

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int HEADS>
Bf16DenseAttnScoreLaunch bindWidth(int width) {
    if constexpr (Implementation::kRingTeacher) {
        if (width == 576) return &launchBf16DenseAttnScore<Implementation, 576, HEADS>;
        return &launchBf16DenseAttnScore<Implementation, 512, HEADS>;
    } else if constexpr (Implementation::kHoisted) {
        if (width == 128) return &launchBf16DenseAttnScore<Implementation, 128, HEADS>;
        return &launchBf16DenseAttnScore<Implementation, 64, HEADS>;
    } else if constexpr (Implementation::kLargeD) {
        switch (width) {
            case 16: return &launchBf16DenseAttnScore<Implementation, 16, HEADS>;
            case 32: return &launchBf16DenseAttnScore<Implementation, 32, HEADS>;
            case 48: return &launchBf16DenseAttnScore<Implementation, 48, HEADS>;
            case 64: return &launchBf16DenseAttnScore<Implementation, 64, HEADS>;
            case 80: return &launchBf16DenseAttnScore<Implementation, 80, HEADS>;
            case 96: return &launchBf16DenseAttnScore<Implementation, 96, HEADS>;
            case 112: return &launchBf16DenseAttnScore<Implementation, 112, HEADS>;
            case 128: return &launchBf16DenseAttnScore<Implementation, 128, HEADS>;
            case 256: return &launchBf16DenseAttnScore<Implementation, 256, HEADS>;
            case 448: return &launchBf16DenseAttnScore<Implementation, 448, HEADS>;
            case 512: return &launchBf16DenseAttnScore<Implementation, 512, HEADS>;
            case 576: return &launchBf16DenseAttnScore<Implementation, 576, HEADS>;
            default: throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured dense width is not instantiated");
        }
    } else {
        return &launchBf16DenseAttnScore<Implementation, 0, HEADS>;
    }
}

template<class Implementation>
Bf16DenseAttnScoreLaunch bindBf16DenseAttnScore(int width, int heads) {
    if constexpr (Implementation::kRingTeacher || Implementation::kHoisted) {
        switch (heads) {
            case 16: return bindWidth<Implementation, 16>(width);
            case 32: return bindWidth<Implementation, 32>(width);
            case 64: return bindWidth<Implementation, 64>(width);
            case 128: return bindWidth<Implementation, 128>(width);
            default: return bindWidth<Implementation, 0>(width);
        }
    } else {
        return bindWidth<Implementation, 0>(width);
    }
}

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

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

template<class Implementation>
const char* Bf16DenseAttnScoreBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16DenseAttnScoreBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16DenseAttnScoreBase<Implementation>::isApplicable(const DenseAttnScoreProblem& problem,
                                                            const Constraints& constraints) const {
    if constexpr (Implementation::kPadding) {
        if (problem.getQueryHeads() != 64 || problem.getHeadDim() != 576 || problem.getRatio() != 1 ||
            problem.getQueryLength() < 512 || problem.getKeyLength() < 512 ||
            problem.getKeyLength() * 576 * 2 > 2147483647ULL) {
            return Result(FTRAIN_STATUS_UNSUPPORTED,
                          "head-padding classifier requires bounded H64 D576 ratio-one rows");
        }
    }

    if constexpr (Implementation::kBpermuteReduce) {
        if (problem.getQueryHeads() != 64 || problem.getHeadDim() != 512) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "bpermute reduction requires H64 D512");
        }
    }

    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKAddress())) &
         7) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 8-byte alignment");
    }

    if ((Implementation::kRingTeacher) && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires gfx936 or gfx938");
    }

    if ((Implementation::kRingTeacher) && ((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
                                            reinterpret_cast<std::uintptr_t>(problem.getKAddress())) &
                                           15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires 16-byte aligned vector operands");
    }
    if (!(((Implementation::kRingTeacher && ((problem.getHeadDim() == 512 || problem.getHeadDim() == 576) &&
                                             (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 ||
                                              problem.getQueryHeads() == 64 || problem.getQueryHeads() == 128)))) ||
          ((Implementation::kStaged && (problem.getQueryHeads() <= 64 && problem.getHeadDim() <= 128)) ||
           (Implementation::kLargeD) ||
           (Implementation::kHoisted && (problem.getQueryHeads() <= 64 && problem.getQueryHeads() % 16 == 0 &&
                                         (problem.getHeadDim() == 64 || problem.getHeadDim() == 128)))))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getBatch() == 0 || problem.getQueryHeads() == 0) {
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
    // Check required scalar addresses here as well as in StorageView validation.
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
    if (!bf16DenseAttnScoreDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!bf16DenseAttnScoreLaunchFits<Implementation>(static_cast<int>(problem.getQueryHeads()),
                                                      static_cast<int>(problem.getHeadDim()),
                                                      static_cast<int>(problem.getKeyLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16DenseAttnScoreBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Bf16DenseAttnScoreBase<Implementation>::configure(const DenseAttnScoreProblem& problem) {
    if constexpr (Implementation::kPadding) {
        groups_          = static_cast<int>((problem.getBatch() * problem.getQueryLength() + 15) / 16);
        workspace_bytes_ = (static_cast<std::uint64_t>(groups_) + 1) * sizeof(int);
    }

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
    if constexpr (!Implementation::kPadding) {
        launch_ = bindBf16DenseAttnScore<Implementation>(head_dim_, query_heads_);
    }
}

template<class Implementation>
void Bf16DenseAttnScoreBase<Implementation>::executeImpl(const Resources& resources) {
    bool launched;
    if constexpr (Implementation::kPadding) {
        launched = launchBf16DenseAttnScorePadding(resources.getStream(), q_, k_, lse_, out_, denom_, batch_,
                                                   query_length_, key_length_, softmax_scale_,
                                                   static_cast<int*>(resources.getWorkspace()), groups_);
    } else {
        launched = launch_(resources.getStream(), q_, k_, lse_, out_, denom_, batch_, query_length_, key_length_,
                           query_heads_, 1, head_dim_, ratio_, softmax_scale_);
    }
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "DenseAttnScore kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16DenseAttnScoreBase<Bf16DenseAttnScoreRingTeacher>;
template class Bf16DenseAttnScoreBase<Bf16DenseAttnScorePadding>;
template class Bf16DenseAttnScoreBase<Bf16DenseAttnScoreRingTeacherBpermute>;
template class Bf16DenseAttnScoreBase<Bf16DenseAttnScore>;
template class Bf16DenseAttnScoreBase<Bf16DenseAttnScoreLargeD>;
template class Bf16DenseAttnScoreBase<Bf16DenseAttnScoreHoisted>;
}  // namespace ftrain
