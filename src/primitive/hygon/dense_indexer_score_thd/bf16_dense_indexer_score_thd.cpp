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
#include "flash_train/primitive/hygon/dense_indexer_score_thd/bf16_dense_indexer_score_thd.hpp"

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

static inline bool dense_mmac_shm_fits(int Hq, int D, int Sk, bool bwd) {
    const int max_lds = dsa_shared_memory_limit();
    if (!bwd) return dense_mmac_shm(Hq, D, Sk, false) <= (size_t)max_lds;
    int Hq_pad = (Hq + 15) & ~15;
    return dense_mmac_shm(Hq, D, Sk, true) <= (size_t)max_lds;
}

static inline bool dense_mmac_storage_fits(int heads, int width, int keys, bool packed) {
    if (heads <= 0 || heads > 128 || width <= 0 || width > 576 || width % 16 || keys <= 0) return false;
    if (!dense_mmac_shm_fits(heads, width, keys, false)) return false;

    if (width > 128 && width != 256 && width != 448 && width != 512 && width != 576) return false;
    return packed || (width <= 128 && heads <= 64) ||
           dsa_shared_memory_fits(dense_mmac_shm_largeD_tpl(heads, width, keys));
}
}  // namespace

template<class Implementation>
bool bf16DenseIndexerScoreThdLaunchFits(int heads, int width, int keys) {
    if (heads <= 0 || heads > 128 || width <= 0 || width > 576 || width % 16 || keys <= 0) return false;
    if (width > 128 && width != 256 && width != 448 && width != 512 && width != 576) return false;
    // Fixed-tile algorithms do not allocate a score row in LDS. Check the
    // selected implementation's footprint, not the historical launch heuristic.
    if constexpr (Implementation::kMultiQuery || Implementation::kTransposed) {
        return dsa_shared_memory_fits(4 * 2048 * sizeof(std::uint16_t));
    } else if constexpr (Implementation::kStaged) {
        return dense_mmac_storage_fits(heads, width, keys, true);
    }
    return false;
}

bool bf16DenseIndexerScoreThdDenseMmacSupported() { return dense_mmac_supported(); }

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int HEADS>
Bf16DenseIndexerScoreThdLaunch bindWidth(int width) {
    switch (width) {
        case 16: return &launchBf16DenseIndexerScoreThd<Implementation, 16, HEADS>;
        case 32: return &launchBf16DenseIndexerScoreThd<Implementation, 32, HEADS>;
        case 48: return &launchBf16DenseIndexerScoreThd<Implementation, 48, HEADS>;
        case 64: return &launchBf16DenseIndexerScoreThd<Implementation, 64, HEADS>;
        case 80: return &launchBf16DenseIndexerScoreThd<Implementation, 80, HEADS>;
        case 96: return &launchBf16DenseIndexerScoreThd<Implementation, 96, HEADS>;
        case 112: return &launchBf16DenseIndexerScoreThd<Implementation, 112, HEADS>;
        case 128: return &launchBf16DenseIndexerScoreThd<Implementation, 128, HEADS>;
        case 256: return &launchBf16DenseIndexerScoreThd<Implementation, 256, HEADS>;
        case 448: return &launchBf16DenseIndexerScoreThd<Implementation, 448, HEADS>;
        case 512: return &launchBf16DenseIndexerScoreThd<Implementation, 512, HEADS>;
        case 576: return &launchBf16DenseIndexerScoreThd<Implementation, 576, HEADS>;
        default: throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured THD width has no symbol");
    }
}

template<class Implementation>
Bf16DenseIndexerScoreThdLaunch bindBf16DenseIndexerScoreThd(int width, int heads) {
    if constexpr (Implementation::kMultiQuery || Implementation::kTransposed) {
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

bool hasInt32Indexing(const DenseIndexerScoreThdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getMaxSeqlenKv() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKDims(), &problem.getWDims(), &problem.getCuSeqlensQDims(),
          &problem.getCuSeqlensKvDims(), &problem.getQCausalOffsetsDims(), &problem.getOutDims(),
          &problem.getDenomDims(), &problem.getRatioDims(), &problem.getSmScaleDims(), &problem.getMaxSeqlenKvDims()}) {
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
const char* Bf16DenseIndexerScoreThdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16DenseIndexerScoreThdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16DenseIndexerScoreThdBase<Implementation>::isApplicable(const DenseIndexerScoreThdProblem& problem,
                                                                  const Constraints& constraints) const {
    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKAddress())) &
         7) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 8-byte alignment");
    }

    if ((!Implementation::kStaged) && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires gfx936 or gfx938");
    }

    if ((!Implementation::kStaged) && ((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
                                        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) |
                                        reinterpret_cast<std::uintptr_t>(problem.getWAddress())) &
                                       15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires 16-byte aligned vector operands");
    }
    if (!(Implementation::kStaged ||
          (((Implementation::kMultiQuery &&
             (problem.getHeadDim() == 128 &&
              (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64))) ||
            (Implementation::kTransposed && problem.getHeadDim() == 128 &&
             (problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64)))))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getQueryHeads() == 0) {
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
        problem.getCuSeqLensQNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCuSeqLensKvNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCausalOffsetsNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getDenomNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getMaxSeqlenKvNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCausalOffsetsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDenomIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getMaxSeqlenKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensQAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensKvAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDenomAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getMaxSeqlenKvAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr selects zero causal offsets; only supplied offsets need address validation.
    if (problem.getCausalOffsetsAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getCausalOffsetsAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKAddress() == nullptr || problem.getWAddress() == nullptr ||
        problem.getCuSeqLensQAddress() == nullptr || problem.getCuSeqLensKvAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getDenomAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getRatioAddress() == nullptr ||
        problem.getMaxSeqlenKvAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.hasSingleKvHead()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the kernel supports one KV head"); }
    if (problem.getTotalK() == 0 || problem.getTotalQ() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty packed query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % kHeadDimAlignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (!bf16DenseIndexerScoreThdDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!bf16DenseIndexerScoreThdLaunchFits<Implementation>(static_cast<int>(problem.getQueryHeads()),
                                                            static_cast<int>(problem.getHeadDim()),
                                                            static_cast<int>(problem.getMaxSeqlenKv()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16DenseIndexerScoreThdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Bf16DenseIndexerScoreThdBase<Implementation>::configure(const DenseIndexerScoreThdProblem& problem) {
    q_              = problem.getQAddress();
    k_              = problem.getKAddress();
    w_              = problem.getWAddress();
    cu_seqlens_q_   = static_cast<const int*>(problem.getCuSeqLensQAddress());
    cu_seqlens_kv_  = static_cast<const int*>(problem.getCuSeqLensKvAddress());
    causal_offsets_ = problem.getCausalOffsetsAddress() == nullptr
                          ? nullptr
                          : static_cast<const int*>(problem.getCausalOffsetsAddress());
    out_            = static_cast<float*>(problem.getOutAddress());
    denom_          = static_cast<float*>(problem.getDenomAddress());
    total_q_        = static_cast<int>(problem.getTotalQ());
    total_k_        = static_cast<int>(problem.getTotalK());
    max_seqlen_kv_  = static_cast<int>(problem.getMaxSeqlenKv());
    query_heads_    = static_cast<int>(problem.getQueryHeads());
    head_dim_       = static_cast<int>(problem.getHeadDim());
    num_sequences_  = static_cast<int>(problem.getNumSequences());
    ratio_          = static_cast<int>(problem.getRatio());
    sm_scale_       = *static_cast<const float*>(problem.getSmScaleAddress());
    launch_         = bindBf16DenseIndexerScoreThd<Implementation>(head_dim_, query_heads_);
}

template<class Implementation>
void Bf16DenseIndexerScoreThdBase<Implementation>::executeImpl(const Resources& resources) {
    const bool launched = launch_(resources.getStream(), q_, k_, w_, out_, denom_, cu_seqlens_q_, cu_seqlens_kv_,
                                  causal_offsets_, total_q_, total_k_, max_seqlen_kv_, query_heads_, /*kv_heads=*/1,
                                  head_dim_, num_sequences_, ratio_, sm_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "DenseIndexerScoreThd kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16DenseIndexerScoreThdBase<Bf16DenseIndexerScoreThd>;
template class Bf16DenseIndexerScoreThdBase<Bf16DenseIndexerScoreThdMultiQuery>;
template class Bf16DenseIndexerScoreThdBase<Bf16DenseIndexerScoreThdTransposed>;
}  // namespace ftrain
