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
#include "flash_train/primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.hpp"

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

static inline bool sparse_attn_fused_supported() { return sparse_mmac_supported(); }
}  // namespace

bool bf16SparseAttnBwdLaunchFits(int width, int columns, int heads, bool fused) {
    if (width != 512 && width != 576) return false;
    if (fused) {
        return dsa_shared_memory_fits(16ULL * (width + 8) * 2 + 2 * 4 * 256 * 2 +
                                      static_cast<std::size_t>(columns) * 4);
    }
    // Packed stats and gradients reuse fixed tiles. Cached indices are
    // staged only up to 8192; larger selections stream from global memory.
    const std::size_t stats =
        16ULL * (width + 24) * 2 +
        ((width == 512 && (columns == 128 || columns == 160 || columns == 512 || columns == 1024)) ? columns * 4 : 0);
    const std::size_t gradient = 5 * 2 * 16 * 66 * 2;
    const std::size_t cached =
        (heads == 16 || heads == 32 || heads == 64 || heads == 128)
            ? 2ULL * 16 * (heads + ((heads == 128 || (heads == 64 && width == 576)) ? 4 : 2)) * 2 +
                  (columns <= 8192 ? columns * 4 : 0)
            : 0;
    return dsa_shared_memory_fits(stats) && dsa_shared_memory_fits(gradient) && dsa_shared_memory_fits(cached);
}

bool bf16SparseAttnBwdSparseAttnFusedSupported() { return sparse_attn_fused_supported(); }

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, int ND, int HEADS, int TOPK, bool COMPACT>
Bf16SparseAttnBwdLaunch bindBwdAccess(int heads, int keys, int selection) {
    if constexpr (Implementation::kFused || Implementation::kPacked || Implementation::kRegisterFused ||
                  Implementation::kPadded || Implementation::kGlmPair) {
        return &launchBf16SparseAttnBwd<Implementation, ND, HEADS, TOPK, COMPACT, true, true>;
    } else {
        const bool raw   = TOPK > 0 || static_cast<std::uint64_t>(selection) * heads * 4 <= 0x80000000ULL;
        const bool raw_k = static_cast<std::uint64_t>(keys) * ND * 16 * sizeof(std::uint16_t) <= 0x80000000ULL;
        if (raw) {
            if (raw_k) return &launchBf16SparseAttnBwd<Implementation, ND, HEADS, TOPK, COMPACT, true, true>;
            return &launchBf16SparseAttnBwd<Implementation, ND, HEADS, TOPK, COMPACT, true, false>;
        }
        if (raw_k) return &launchBf16SparseAttnBwd<Implementation, ND, HEADS, TOPK, COMPACT, false, true>;
        return &launchBf16SparseAttnBwd<Implementation, ND, HEADS, TOPK, COMPACT, false, false>;
    }
}

template<class Implementation, int ND, int HEADS, bool COMPACT = false>
Bf16SparseAttnBwdLaunch bindBwdColumns(int heads, int keys, int selection) {
    if constexpr (Implementation::kPadded || Implementation::kGlmPair) {
        return bindBwdAccess<Implementation, ND, HEADS, 2048, COMPACT>(heads, keys, selection);
    } else if constexpr (Implementation::kCompact || Implementation::kGrouped) {
        return bindBwdAccess<Implementation, ND, HEADS, 1152, COMPACT>(heads, keys, selection);
    } else if constexpr (Implementation::kCompactValues) {
        return bindBwdAccess<Implementation, ND, HEADS, 0, COMPACT>(heads, keys, selection);
    } else {
        switch (selection) {
#define BWD_COLUMN(K) \
    case K: return bindBwdAccess<Implementation, ND, HEADS, K, COMPACT>(heads, keys, selection)
            BWD_COLUMN(64);
            BWD_COLUMN(128);
            BWD_COLUMN(160);
            BWD_COLUMN(256);
            BWD_COLUMN(512);
            BWD_COLUMN(1024);
            BWD_COLUMN(1152);
            BWD_COLUMN(2048);
#undef BWD_COLUMN
            default: return bindBwdAccess<Implementation, ND, HEADS, 0, COMPACT>(heads, keys, selection);
        }
    }
}

template<class Implementation, int ND>
Bf16SparseAttnBwdLaunch bindBwdHeads(int heads, int keys, int selection) {
    if constexpr (Implementation::kPadded || Implementation::kGlmLocal || Implementation::kLocalFusion ||
                  Implementation::kCompact) {
        return bindBwdColumns<Implementation, ND, 64>(heads, keys, selection);
    } else if constexpr (Implementation::kGlmPair) {
        return bindBwdColumns<Implementation, ND, 32>(heads, keys, selection);
    } else if constexpr (Implementation::kCompactValues) {
        switch (heads) {
#define BWD_HEAD(H) \
    case H: return bindBwdColumns<Implementation, ND, H>(heads, keys, selection)
            BWD_HEAD(8);
            BWD_HEAD(16);
            BWD_HEAD(32);
            BWD_HEAD(64);
            BWD_HEAD(128);
#undef BWD_HEAD
        }
    } else if constexpr (Implementation::kCached || Implementation::kRegisterFused || Implementation::kSplit) {
        if (heads == 64) return bindBwdColumns<Implementation, ND, 64>(heads, keys, selection);
        if (heads == 128) return bindBwdColumns<Implementation, ND, 128>(heads, keys, selection);
        return bindBwdColumns<Implementation, ND, 0>(heads, keys, selection);
    } else {
        return bindBwdColumns<Implementation, ND, 0>(heads, keys, selection);
    }
    throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured backward has no head specialization");
}

template<class Implementation>
Bf16SparseAttnBwdLaunch bindBf16SparseAttnBwd(int heads, int width, int value_width, int keys, int selection) {
    if constexpr (Implementation::kFused) {
        if (value_width != width) return bindBwdColumns<Implementation, 36, 0, true>(heads, keys, selection);
        if (width == 576) return bindBwdHeads<Implementation, 36>(heads, keys, selection);
        return bindBwdHeads<Implementation, 32>(heads, keys, selection);
    } else if constexpr (Implementation::kCompactValues) {
        return bindBwdHeads<Implementation, 36>(heads, keys, selection);
    } else if constexpr (Implementation::kPacked) {
        if (width == 576) return bindBwdHeads<Implementation, 36>(heads, keys, selection);
        return bindBwdHeads<Implementation, 32>(heads, keys, selection);
    } else {
        return bindBwdHeads<Implementation, 32>(heads, keys, selection);
    }
}

bool isGfx936Or938(FTrainDeviceId device) {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, static_cast<int>(device)) != hipSuccess) { return false; }
    return (std::strncmp(props.gcnArchName, "gfx936", 6) == 0 || std::strncmp(props.gcnArchName, "gfx938", 6) == 0) &&
           (props.gcnArchName[6] == '\0' || props.gcnArchName[6] == ':');
}

bool hasInt32Indexing(const SparseAttnBwdProblem& problem) noexcept {
    for (const auto* dims : {&problem.getQDims(), &problem.getKvDims(), &problem.getOutDims(), &problem.getDoutDims(),
                             &problem.getLseDims(), &problem.getAttnSinkDims(), &problem.getTopkIndicesDims(),
                             &problem.getTopkLengthDims(), &problem.getDqDims(), &problem.getDkvDims(),
                             &problem.getDsinkDims(), &problem.getSoftmaxScaleDims()}) {
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

bool supportsShape(const SparseAttnBwdProblem& problem) noexcept {
    return problem.getValueDim() == problem.getHeadDim() || problem.getValueDim() == 512;
}

constexpr std::uint64_t kMaxHeadDim      = 576;
constexpr std::uint64_t kCompactValueDim = 512;
// Completion, locality rejection, valid-key count, and padded-head rejection.
constexpr std::uint64_t kCounterCount = 4;

}  // namespace

template<class Implementation>
const char* Bf16SparseAttnBwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16SparseAttnBwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16SparseAttnBwdBase<Implementation>::isApplicable(const SparseAttnBwdProblem& problem,
                                                           const Constraints& constraints) const {
    // These kernels stage/cast vectors; element alignment alone is insufficient.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDoutAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDkvAddress())) &
         15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 16-byte alignment");
    }

    if ((Implementation::kRegisterFused) && !isGfx936Or938(constraints.getDeviceId())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm requires gfx936 or gfx938");
    }

    const auto T = problem.getTokens();
    const auto H = problem.getHeads();
    const auto D = problem.getHeadDim();
    const auto V = problem.getValueDim();
    const auto S = problem.getKvTokens();
    const auto K = problem.getSelectionWidth();
    const bool split = D == 512 && (K >= 256 || (T >= 256 && K >= 128 && (H == 64 || H == 128)));
    const bool cached = problem.usesCachedBackward() &&
                        (Implementation::kCompactValues
                             ? D == 576 && V == 512 && K >= 128 && (H == 8 || H == 16 || H == 32 || H == 64 || H == 128)
                             : split && (H == 64 || H == 128));
    const bool local = D == 512 && H == 64 && T >= 2048 && S >= K && K <= 8192 &&
                       T * K <= 2147483647ULL && (K == 128 || K == 160 || K >= 1024);
    const bool grouped = T == 4096 && H == 64 && D == 512 && S == 5120 && K == 1152;
    const bool glm_pair = D == 576 && V == 512 && H == 32 && (T == 4096 || T == 4097) &&
                          S == 4096 && K == 2048;
    const bool padded = D == 576 && V == 512 && H == 64 && (T == 4096 || T == 4097) &&
                        S == 4096 && K == 2048;
    const bool glm_local = D == 576 && V == 512 && H == 64 && T >= 2048 && K >= 1024 &&
                           K <= 8192 && S >= K && T * K <= 2147483647ULL;
    bool supported = false;
    if constexpr (Implementation::kFused) {
        supported = true;
    } else if constexpr (Implementation::kRegisterFused) {
        supported = D == 512 && (H == 64 || H == 128) && T >= 256 && S % 8 == 0 && (K == 128 || K == 160);
    } else if constexpr (Implementation::kPacked) {
        supported = (D == 512 || D == 576) && V == D && (!Implementation::kSplit || split);
    } else if constexpr (Implementation::kCached || Implementation::kCachedLowMemory) {
        supported = cached && (Implementation::kGlobalIndices || K <= 8192);
        if constexpr (Implementation::kPadded) supported = supported && padded;
        if constexpr (Implementation::kGlmPair) supported = supported && glm_pair;
        if constexpr (Implementation::kGlmLocal) supported = supported && glm_local;
        if constexpr (Implementation::kCompact) supported = supported && grouped && problem.getTopkLengthAddress();
        if constexpr (Implementation::kLocalFusion) {
            supported = supported && local && (Implementation::kKeyPacked ? S % 8 == 0 : S % 8 != 0);
        }
        if constexpr (Implementation::kGrouped) supported = supported && grouped;
        if constexpr (Implementation::kNoPrefetch) supported = supported && D == 512 && H == 128;
    }
    if (!supported) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

    if (problem.getHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires full or 512-wide value channels");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDoutNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAttnSinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getDqNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDkvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getDsinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDoutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnSinkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkLengthIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDqIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDkvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getDsinkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDoutAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnSinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDkvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getDsinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr selects the full top-k extent; only supplied lengths need address validation.
    if (problem.getTopkLengthAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getTopkLengthAddress()) % alignof(std::int32_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKvAddress() == nullptr || problem.getOutAddress() == nullptr ||
        problem.getDoutAddress() == nullptr || problem.getLseAddress() == nullptr ||
        problem.getAttnSinkAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getDqAddress() == nullptr || problem.getDkvAddress() == nullptr ||
        problem.getDsinkAddress() == nullptr || problem.getSoftmaxScaleAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (problem.getTokens() == 0) { return Result(FTRAIN_STATUS_UNSUPPORTED, "an empty query has no kernel work"); }
    if (problem.getHeadDim() < kCompactValueDim || problem.getHeadDim() > kMaxHeadDim ||
        problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "Q/K head dimension is outside the kernel's staged width");
    }
    if (!bf16SparseAttnBwdSparseAttnFusedSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device does not expose the MMAC sparse-attention path this kernel needs");
    }
    if (!bf16SparseAttnBwdLaunchFits(static_cast<int>(problem.getHeadDim()),
                                     static_cast<int>(problem.getSelectionWidth()),
                                     static_cast<int>(problem.getHeads()), Implementation::kFused)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16SparseAttnBwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return workspace_bytes_;
}

template<class Implementation>
void Bf16SparseAttnBwdBase<Implementation>::configure(const SparseAttnBwdProblem& problem) {
    q_                = problem.getQAddress();
    kv_               = problem.getKvAddress();
    out_              = problem.getOutAddress();
    dout_             = problem.getDoutAddress();
    lse_              = static_cast<const float*>(problem.getLseAddress());
    attn_sink_        = static_cast<const float*>(problem.getAttnSinkAddress());
    topk_indices_     = static_cast<const int*>(problem.getTopkIndicesAddress());
    topk_length_      = static_cast<const int*>(problem.getTopkLengthAddress());
    dq_               = problem.getDqAddress();
    dkv_              = problem.getDkvAddress();
    dsink_            = static_cast<float*>(problem.getDsinkAddress());
    tokens_           = static_cast<int>(problem.getTokens());
    heads_            = static_cast<int>(problem.getHeads());
    head_dim_         = static_cast<int>(problem.getHeadDim());
    backward_width_   = static_cast<int>(problem.getValueDim());
    kv_tokens_        = static_cast<int>(problem.getKvTokens());
    selection_width_  = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_    = *static_cast<const float*>(problem.getSoftmaxScaleAddress());

    launch_ = bindBf16SparseAttnBwd<Implementation>(heads_, head_dim_, backward_width_, kv_tokens_, selection_width_);

    // Preserve the cache-line alignment of the original standalone packed-stat allocation.
    workspace_bytes_   = 0;
    dkv_float_offset_  = workspace_bytes_;
    workspace_bytes_  += static_cast<std::uint64_t>(kv_tokens_) * static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += kCounterCount * sizeof(int);
    if constexpr (!Implementation::kFused) {
        workspace_bytes_              = (workspace_bytes_ + 255) & ~std::uint64_t{255};
        packed_stats_offset_          = workspace_bytes_;
        const std::uint64_t rows = Implementation::kCachedLowMemory ? (tokens_ + 1) / 2 : tokens_;
        const std::uint64_t columns = static_cast<std::uint64_t>(selection_width_) * heads_ +
                                      (Implementation::kPadded ? kv_tokens_ + 36ULL * 4 * 64 * 4 : 0);
        workspace_bytes_             += rows * columns * sizeof(std::int32_t);
    }
}

template<class Implementation>
void Bf16SparseAttnBwdBase<Implementation>::executeImpl(const Resources& resources) {
    float* const dkv_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dkv_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    // The upstream launchers initialize dkv_float themselves (fused with
    // classification for fused2); do not add a redundant memset here.
    // The register-fused algorithm initializes both buffers in its stream-ordered
    // initializer. Repeating the resets here adds two graph nodes to every execution.
    if (!Implementation::kRegisterFused) {
        bf16SparseAttnBwdZeroWorkspace(resources.getStream(), done_count, kCounterCount * sizeof(int));
        bf16SparseAttnBwdZeroWorkspace(resources.getStream(), dsink_,
                                       static_cast<std::uint64_t>(heads_) * sizeof(float));
    }

    void* packed_stats = nullptr;
    if constexpr (!Implementation::kFused) {
        packed_stats = static_cast<unsigned char*>(resources.getWorkspace()) + packed_stats_offset_;
    }
    if (!launch_(resources.getStream(), q_, kv_, out_, dout_, lse_, attn_sink_, topk_indices_, topk_length_, dq_,
                 dkv_float, dsink_, dkv_, done_count, packed_stats, tokens_, heads_, head_dim_, backward_width_,
                 kv_tokens_, selection_width_, softmax_scale_)) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured backward failed to submit its fixed kernels");
    }
}

template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwd>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdPacked>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCached>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedLowMemory>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdRegisterFused>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdPackedSplit>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedNoPrefetch>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedLocal>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedLocalAligned>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGrouped>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedCompact>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlmPair>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlmLocal>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedPadded>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlobalIndices>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedNoPrefetchGlobalIndices>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlm>;
template class Bf16SparseAttnBwdBase<Bf16SparseAttnBwdCachedGlmGlobalIndices>;
}  // namespace ftrain
