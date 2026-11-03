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
#include "flash_train/primitive/hygon/indexer_bwd_dense_thd/bf16_indexer_bwd_dense_thd.hpp"

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
}  // namespace

bool bf16IndexerBwdDenseThdLaunchFits(int heads, int width, int columns, int queries) {
    (void)queries;
    return dense_mmac_shm_fits(heads, width, columns, true);
}

bool bf16IndexerBwdDenseThdDenseMmacSupported() { return dense_mmac_supported(); }

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, bool PAIR, bool PROOF>
Bf16IndexerBwdDenseThdLaunch bindWidth(int width) {
    if constexpr (Implementation::kTwoWave) {
        return &launchBf16IndexerBwdDenseThd<Implementation, 128, PAIR, PROOF>;
    } else {
        switch (width) {
            case 16: return &launchBf16IndexerBwdDenseThd<Implementation, 16, PAIR, PROOF>;
            case 32: return &launchBf16IndexerBwdDenseThd<Implementation, 32, PAIR, PROOF>;
            case 48: return &launchBf16IndexerBwdDenseThd<Implementation, 48, PAIR, PROOF>;
            case 64: return &launchBf16IndexerBwdDenseThd<Implementation, 64, PAIR, PROOF>;
            case 80: return &launchBf16IndexerBwdDenseThd<Implementation, 80, PAIR, PROOF>;
            case 96: return &launchBf16IndexerBwdDenseThd<Implementation, 96, PAIR, PROOF>;
            case 112: return &launchBf16IndexerBwdDenseThd<Implementation, 112, PAIR, PROOF>;
            case 128: return &launchBf16IndexerBwdDenseThd<Implementation, 128, PAIR, PROOF>;
            default:
                throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured dense indexer width is not instantiated");
        }
    }
}

template<class Implementation>
Bf16IndexerBwdDenseThdLaunch bindBf16IndexerBwdDenseThd(int width) {
    return bindWidth<Implementation, Implementation::kPair, Implementation::kProof>(width);
}

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const IndexerBwdDenseThdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims : {&problem.getIndexQDims(), &problem.getWeightsDims(), &problem.getIndexKDims(),
                             &problem.getAttnScoreDims(), &problem.getAttnL1normDims(), &problem.getIndexScoreDims(),
                             &problem.getIndexLseDims(), &problem.getCuSeqlensQDims(), &problem.getCuSeqlensKDims(),
                             &problem.getQCausalOffsetsDims(), &problem.getGradIndexScoreDims(), &problem.getDqDims(),
                             &problem.getDwDims(), &problem.getDkDims(), &problem.getSmScaleDims(),
                             &problem.getLossCoeffDims(), &problem.getGradLossDims(), &problem.getRatioDims()}) {
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

constexpr std::uint64_t kCounterCount = 1;

}  // namespace

template<class Implementation>
const char* Bf16IndexerBwdDenseThdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16IndexerBwdDenseThdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16IndexerBwdDenseThdBase<Implementation>::isApplicable(const IndexerBwdDenseThdProblem& problem,
                                                                const Constraints& constraints) const {
    if constexpr (Implementation::kTwoWave) {
        const bool paired = problem.getRatio() > 1 && problem.getRatio() % 2 == 0 && problem.getNumSequences() == 1;
        if (Implementation::kPair != paired) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "two-wave schedule does not match the paired-row coverage");
        }
    }
    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDkAddress())) &
         7) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 8-byte alignment");
    }

    if ((Implementation::kTwoWave) && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires gfx936 or gfx938");
    }

    if ((Implementation::kTwoWave) && ((reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) |
                                        reinterpret_cast<std::uintptr_t>(problem.getWeightsAddress()) |
                                        reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) |
                                        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) |
                                        reinterpret_cast<std::uintptr_t>(problem.getDwAddress()) |
                                        reinterpret_cast<std::uintptr_t>(problem.getDkAddress())) &
                                       15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires 16-byte aligned vector operands");
    }
    if (!(Implementation::kFused ||
          (problem.getHeads() == 64 && problem.getHeadDim() == 128 && problem.getRatio() > 0))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    // Validate type, layout and alignment even when the output aliases its input.
    if (problem.getGradIndexScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradIndexScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "in-place score gradient requires continuous FP32 storage");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getIndexQNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getWeightsNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getIndexKNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getAttnScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAttnL1NormNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getIndexScoreNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getIndexLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getCuSeqLensQNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCuSeqLensKNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCausalOffsetsNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getDqNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDwNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDkNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getSmScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getLossCoeffNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getGradLossNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getIndexQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getWeightsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnL1NormIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensKIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCausalOffsetsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDqIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDwIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSmScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLossCoeffIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradLossIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getIndexQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWeightsAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexKAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnL1NormAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradIndexScoreAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensQAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensKAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDwAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDkAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSmScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLossCoeffAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradLossAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr selects zero causal offsets; only supplied offsets need address validation.
    if (problem.getCausalOffsetsAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getCausalOffsetsAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getIndexQAddress() == nullptr || problem.getWeightsAddress() == nullptr ||
        problem.getIndexKAddress() == nullptr || problem.getAttnScoreAddress() == nullptr ||
        problem.getAttnL1NormAddress() == nullptr || problem.getIndexScoreAddress() == nullptr ||
        problem.getIndexLseAddress() == nullptr || problem.getCuSeqLensQAddress() == nullptr ||
        problem.getCuSeqLensKAddress() == nullptr || problem.getDqAddress() == nullptr ||
        problem.getDwAddress() == nullptr || problem.getDkAddress() == nullptr ||
        problem.getSmScaleAddress() == nullptr || problem.getRatioAddress() == nullptr ||
        problem.getLossCoeffAddress() == nullptr || problem.getGradLossAddress() == nullptr ||
        problem.getGradIndexScoreAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.isInPlace()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "grad_index_score must alias index_score"); }
    if (problem.getTotalK() == 0 || problem.getTotalQ() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty packed query or key length has no kernel work");
    }
    if (problem.getHeadDim() == 0 || problem.getHeadDim() > 128 || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "head dimension is outside the kernel's MMAC tile coverage");
    }
    if (problem.getHeads() == 0 || problem.getHeads() > 64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the head count is outside the kernel's tile coverage");
    }
    if (!bf16IndexerBwdDenseThdDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!bf16IndexerBwdDenseThdLaunchFits(static_cast<int>(problem.getHeads()), static_cast<int>(problem.getHeadDim()),
                                          static_cast<int>(problem.getMaxSeqlenKv()), 0)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16IndexerBwdDenseThdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Bf16IndexerBwdDenseThdBase<Implementation>::configure(const IndexerBwdDenseThdProblem& problem) {
    index_q_        = problem.getIndexQAddress();
    weights_        = problem.getWeightsAddress();
    index_k_        = problem.getIndexKAddress();
    attn_score_     = static_cast<const float*>(problem.getAttnScoreAddress());
    attn_l1norm_    = static_cast<const float*>(problem.getAttnL1NormAddress());
    index_lse_      = static_cast<const float*>(problem.getIndexLseAddress());
    cu_seqlens_q_   = static_cast<const int*>(problem.getCuSeqLensQAddress());
    cu_seqlens_k_   = static_cast<const int*>(problem.getCuSeqLensKAddress());
    causal_offsets_ = problem.getCausalOffsetsAddress() == nullptr
                          ? nullptr
                          : static_cast<const int*>(problem.getCausalOffsetsAddress());
    // The kernel reads the predictions and writes their gradient through one
    // buffer, which the Problem guarantees to be the same address.
    score_gradient_ = static_cast<float*>(problem.getGradIndexScoreAddress());
    dq_             = problem.getDqAddress();
    dw_             = problem.getDwAddress();
    dk_             = problem.getDkAddress();
    total_q_        = static_cast<int>(problem.getTotalQ());
    total_k_        = static_cast<int>(problem.getTotalK());
    max_seqlen_kv_  = static_cast<int>(problem.getMaxSeqlenKv());
    heads_          = static_cast<int>(problem.getHeads());
    head_dim_       = static_cast<int>(problem.getHeadDim());
    num_sequences_  = static_cast<int>(problem.getNumSequences());
    ratio_          = static_cast<int>(problem.getRatio());
    sm_scale_       = *static_cast<const float*>(problem.getSmScaleAddress());
    loss_coeff_     = *static_cast<const float*>(problem.getLossCoeffAddress());
    grad_loss_      = *static_cast<const float*>(problem.getGradLossAddress());
    launch_         = bindBf16IndexerBwdDenseThd<Implementation>(head_dim_);

    workspace_bytes_    = 0;
    dk_float_offset_    = workspace_bytes_;
    workspace_bytes_   += static_cast<std::uint64_t>(total_k_) * static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += kCounterCount * sizeof(int);
}

template<class Implementation>
void Bf16IndexerBwdDenseThdBase<Implementation>::executeImpl(const Resources& resources) {
    float* const dk_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dk_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    const std::uint64_t dk_float_bytes =
        static_cast<std::uint64_t>(total_k_) * static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    bf16IndexerBwdDenseThdZeroWorkspace(resources.getStream(), dk_float, dk_float_bytes);
    bf16IndexerBwdDenseThdZeroWorkspace(resources.getStream(), done_count, kCounterCount * sizeof(int));

    const bool launched = launch_(resources.getStream(), index_q_, weights_, index_k_, attn_score_, attn_l1norm_,
                                  score_gradient_, index_lse_, dq_, dw_, dk_, dk_float, done_count, cu_seqlens_q_,
                                  cu_seqlens_k_, causal_offsets_, total_q_, total_k_, max_seqlen_kv_, heads_, head_dim_,
                                  num_sequences_, ratio_, sm_scale_, loss_coeff_, grad_loss_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "IndexerBwdDenseThd kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16IndexerBwdDenseThdBase<Bf16IndexerBwdDenseThd>;
template class Bf16IndexerBwdDenseThdBase<Bf16IndexerBwdDenseThdTwoWave>;
template class Bf16IndexerBwdDenseThdBase<Bf16IndexerBwdDenseThdTwoWavePair>;

}  // namespace ftrain
