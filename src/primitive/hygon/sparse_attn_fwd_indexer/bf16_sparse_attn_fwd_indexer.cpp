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
#include "flash_train/primitive/hygon/sparse_attn_fwd_indexer/bf16_sparse_attn_fwd_indexer.hpp"

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

static inline bool sparse_mmac_supported() {
    int dev = 0;
    hipDeviceProp_t_v2 prop{};
    if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties_v2(&prop, dev) != hipSuccess) return false;
    return strstr(prop.gcnArchName, "gfx936") != nullptr || strstr(prop.gcnArchName, "gfx938") != nullptr;
}

static bool sparse_attn_fused_supported() { return sparse_mmac_supported(); }
}  // namespace

template<class Implementation>
bool bf16SparseAttnFwdIndexerLaunchFits(int width, int columns) {
    if (width != 512 && width != 576) return false;
    const int slots = Implementation::kTileIndex ? 32 : columns;
    return dsa_shared_memory_fits(16ULL * (width + 8) * 2 + static_cast<std::size_t>(slots) * 4 + 4 * 256 * 2);
}

bool bf16SparseAttnFwdIndexerSparseAttnFusedSupported() { return sparse_attn_fused_supported(); }

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int ND>
Bf16SparseAttnFwdIndexerLaunch bindColumns(int columns) {
    if constexpr (Implementation::kWindow640) {
        return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 640>;
    } else if constexpr (Implementation::kTileIndex) {
        if constexpr (ND == 36) return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 2048>;
        if (columns == 1024) return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 1024>;
        return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 1152>;
    } else if constexpr (!Implementation::kFused) {
        return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 0>;
    } else {
        switch (columns) {
            case 64: return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 64>;
            case 128: return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 128>;
            case 256: return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 256>;
            case 512: return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 512>;
            case 1024: return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 1024>;
            case 2048: return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 2048>;
            default: return &launchBf16SparseAttnFwdIndexer<Implementation, ND, 0>;
        }
    }
}

template<class Implementation>
Bf16SparseAttnFwdIndexerLaunch bindBf16SparseAttnFwdIndexer(int width, int columns) {
    if (width == 576) return bindColumns<Implementation, 36>(columns);
    return bindColumns<Implementation, 32>(columns);
}

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const SparseAttnFwdIndexerProblem& problem) noexcept {
    if (problem.getIndexerTopk() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKvDims(), &problem.getAttnSinkDims(), &problem.getTopkIndicesDims(),
          &problem.getTopkLengthDims(), &problem.getOutDims(), &problem.getMaxLogitsDims(), &problem.getLseDims(),
          &problem.getLseIndexerDims(), &problem.getSoftmaxScaleDims(), &problem.getIndexerTopkDims()}) {
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

bool supportsValueWidth(const SparseAttnFwdIndexerProblem& problem) noexcept { return problem.getValueDim() == 512; }

constexpr std::uint64_t kMaxHeadDim = 576;
constexpr std::uint64_t kValueDim   = 512;

}  // namespace

template<class Implementation>
const char* Bf16SparseAttnFwdIndexerBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16SparseAttnFwdIndexerBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16SparseAttnFwdIndexerBase<Implementation>::isApplicable(const SparseAttnFwdIndexerProblem& problem,
                                                                  const Constraints& constraints) const {
    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getOutAddress())) &
         15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 16-byte alignment");
    }

    if (!Implementation::kFused && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MLA64 requires gfx936/gfx938");
    }

    if (!(Implementation::kFused ||
          ((problem.getKvTokens() > 0 && problem.getKvTokens() * problem.getHeadDim() * 2 <= 2147483647ULL &&
            problem.getTokens() <= 2147483647ULL && problem.getIndexerTopk() % 64 == 0) &&
           ((Implementation::kMla64 && ((problem.getHeads() == 64 || problem.getHeads() == 128) &&
                                        ((problem.getHeadDim() == 512 &&
                                          (problem.getSelectionWidth() == 128 || problem.getSelectionWidth() == 256 ||
                                           problem.getSelectionWidth() == 512 || problem.getSelectionWidth() == 640 ||
                                           problem.getSelectionWidth() == 1024 || problem.getSelectionWidth() == 1152 ||
                                           problem.getSelectionWidth() == 2048)) ||
                                         (problem.getHeadDim() == 576 && problem.getSelectionWidth() == 2048)))) ||
            (Implementation::kMlaMasked &&
             ((problem.getHeads() == 64 || problem.getHeads() == 128) && problem.getHeadDim() == 512 &&
              (problem.getSelectionWidth() == 1024 || problem.getSelectionWidth() == 1152 ||
               problem.getSelectionWidth() == 2048))) ||
            (Implementation::kMlaTail160 && ((problem.getHeads() == 64 || problem.getHeads() == 128) &&
                                             problem.getHeadDim() == 512 && problem.getSelectionWidth() == 160)) ||
            (Implementation::kMlaTail256 &&
             ((problem.getHeads() == 64 || problem.getHeads() == 128) && problem.getHeadDim() == 512 &&
              problem.getSelectionWidth() == 256 && problem.getTopkLengthAddress() == nullptr)) ||
            (Implementation::kMlaPartial2 &&
             ((problem.getHeads() == 8 || problem.getHeads() == 16 || problem.getHeads() == 32) &&
              problem.getHeadDim() == 576 && problem.getSelectionWidth() == 2048)) ||
            (Implementation::kMlaPartial4 && (problem.getHeads() > 0 && problem.getHeads() < 64 &&
                                              problem.getHeadDim() == 576 && problem.getSelectionWidth() == 2048)))))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if constexpr (Implementation::kTileIndex) {
        if (!((problem.getHeadDim() == 512 &&
               (problem.getSelectionWidth() == 1024 || problem.getSelectionWidth() == 1152)) ||
              (problem.getHeadDim() == 576 && problem.getSelectionWidth() == 2048))) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "tiled index kernel is not instantiated for this D/K");
        }
    }
    if constexpr (Implementation::kWindow640) {
        if (problem.getHeadDim() != 512 || problem.getSelectionWidth() != 640 || problem.getIndexerTopk() != 512) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "window kernel requires D512, K640 and indexer K512");
        }
    }

    if (problem.getHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getAttnSinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getMaxLogitsNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getLseIndexerNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getIndexerTopkNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnSinkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkLengthIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getMaxLogitsIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexerIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getIndexerTopkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnSinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getMaxLogitsAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseIndexerAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getIndexerTopkAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr selects the full top-k extent; only supplied lengths need address validation.
    if (problem.getTopkLengthAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getTopkLengthAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKvAddress() == nullptr ||
        problem.getAttnSinkAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getMaxLogitsAddress() == nullptr ||
        problem.getLseAddress() == nullptr || problem.getLseIndexerAddress() == nullptr ||
        problem.getSoftmaxScaleAddress() == nullptr || problem.getIndexerTopkAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.hasNonEmptyInputs()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the request has no kernel work"); }
    if (!problem.hasSingleKvHead()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the kernel supports one KV head"); }
    if (problem.getHeadDim() < kValueDim || problem.getHeadDim() > kMaxHeadDim || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "Q/K head dimension is outside the kernel's staged width");
    }
    if (!supportsValueWidth(problem)) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the value width must be 512"); }
    if (!problem.hasValidSelection()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "indexer_topk must lie in (0, selection width]");
    }
    if (!bf16SparseAttnFwdIndexerSparseAttnFusedSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device does not expose the MMAC sparse-attention path this kernel needs");
    }
    if (!bf16SparseAttnFwdIndexerLaunchFits<Implementation>(static_cast<int>(problem.getHeadDim()),
                                            static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16SparseAttnFwdIndexerBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Bf16SparseAttnFwdIndexerBase<Implementation>::configure(const SparseAttnFwdIndexerProblem& problem) {
    q_               = problem.getQAddress();
    kv_              = problem.getKvAddress();
    attn_sink_       = static_cast<const float*>(problem.getAttnSinkAddress());
    topk_indices_    = static_cast<const int*>(problem.getTopkIndicesAddress());
    topk_length_     = static_cast<const int*>(problem.getTopkLengthAddress());
    out_             = problem.getOutAddress();
    max_logits_      = static_cast<float*>(problem.getMaxLogitsAddress());
    lse_             = static_cast<float*>(problem.getLseAddress());
    lse_indexer_     = static_cast<float*>(problem.getLseIndexerAddress());
    indexer_topk_    = static_cast<int>(problem.getIndexerTopk());
    tokens_          = static_cast<int>(problem.getTokens());
    heads_           = static_cast<int>(problem.getHeads());
    head_dim_        = static_cast<int>(problem.getHeadDim());
    value_dim_       = static_cast<int>(problem.getValueDim());
    kv_tokens_       = static_cast<int>(problem.getKvTokens());
    selection_width_ = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_   = *static_cast<const float*>(problem.getSoftmaxScaleAddress());
    launch_          = bindBf16SparseAttnFwdIndexer<Implementation>(head_dim_, selection_width_);
}

template<class Implementation>
void Bf16SparseAttnFwdIndexerBase<Implementation>::executeImpl(const Resources& resources) {
    const bool launched = launch_(resources.getStream(), q_, kv_, attn_sink_, topk_indices_, out_, max_logits_, lse_,
                                  lse_indexer_, topk_length_, indexer_topk_, tokens_, heads_, head_dim_, value_dim_,
                                  kv_tokens_, selection_width_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnFwdIndexer kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexer>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerTileIndex>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerWindow640>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMla64>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaMasked>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaTail160>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaTail256>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaPartial2>;
template class Bf16SparseAttnFwdIndexerBase<Bf16SparseAttnFwdIndexerMlaPartial4>;
}  // namespace ftrain
