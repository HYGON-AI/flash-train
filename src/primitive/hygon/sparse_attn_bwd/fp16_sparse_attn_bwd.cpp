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
#include "flash_train/primitive/hygon/sparse_attn_bwd/fp16_sparse_attn_bwd.hpp"

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

bool fp16SparseAttnBwdLaunchFits(int width, int columns) {
    if (width != 512 && width != 576) return false;
    return dsa_shared_memory_fits((16ULL * (width + 8) + 5120) * 2 + static_cast<std::size_t>(columns) * 4);
}

bool fp16SparseAttnBwdSparseAttnFusedSupported() { return sparse_attn_fused_supported(); }

namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
Fp16SparseAttnBwdLaunch bindFp16SparseAttnBwd(int width, int value_width) {
    if (value_width != width) return &launchFp16SparseAttnBwd<576, true>;
    if (width == 576) return &launchFp16SparseAttnBwd<576, false>;
    return &launchFp16SparseAttnBwd<512, false>;
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
// The completion counter plus the two internal dispatch counters the large-D
// variant reads and writes.
constexpr std::uint64_t kCounterCount = 3;

}  // namespace

const char* Fp16SparseAttnBwd::getName() const noexcept { return "Fp16SparseAttnBwd"; }

std::unique_ptr<PrimitiveBase> Fp16SparseAttnBwd::clone() const { return std::make_unique<Fp16SparseAttnBwd>(*this); }

Result Fp16SparseAttnBwd::isApplicable(const SparseAttnBwdProblem& problem, const Constraints&) const {
    // dsa_copy_as stages complete vectors, not just individual elements.
    if (((reinterpret_cast<std::uintptr_t>(problem.getQAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDoutAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDqAddress()) |
          reinterpret_cast<std::uintptr_t>(problem.getDkvAddress())) & 15) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "MMAC vector operands require 16-byte alignment");
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

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getDoutNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAttnSinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getDqNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getDkvNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
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
    if (!fp16SparseAttnBwdSparseAttnFusedSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device does not expose the MMAC sparse-attention path this kernel needs");
    }
    if (!fp16SparseAttnBwdLaunchFits(static_cast<int>(problem.getHeadDim()),
                                     static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Fp16SparseAttnBwd::getRequiredWorkspaceBytes() const noexcept { return workspace_bytes_; }

void Fp16SparseAttnBwd::configure(const SparseAttnBwdProblem& problem) {
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
    launch_           = bindFp16SparseAttnBwd(head_dim_, backward_width_);

    // Each scratch region contains four-byte elements, so adjacent regions
    // retain the required alignment without padding.
    workspace_bytes_   = 0;
    dkv_float_offset_  = workspace_bytes_;
    workspace_bytes_  += static_cast<std::uint64_t>(kv_tokens_) * static_cast<std::uint64_t>(head_dim_) * sizeof(float);
    done_count_offset_  = workspace_bytes_;
    workspace_bytes_   += kCounterCount * sizeof(int);
}

void Fp16SparseAttnBwd::executeImpl(const Resources& resources) {
    float* const dkv_float =
        reinterpret_cast<float*>(static_cast<unsigned char*>(resources.getWorkspace()) + dkv_float_offset_);
    int* const done_count =
        reinterpret_cast<int*>(static_cast<unsigned char*>(resources.getWorkspace()) + done_count_offset_);
    // The upstream launchers initialize dkv_float themselves (fused with
    // classification for fused2); do not add a redundant memset here.
    fp16SparseAttnBwdZeroWorkspace(resources.getStream(), done_count, kCounterCount * sizeof(int));
    fp16SparseAttnBwdZeroWorkspace(resources.getStream(), dsink_, static_cast<std::uint64_t>(heads_) * sizeof(float));

    const bool launched = launch_(resources.getStream(), q_, kv_, out_, dout_, lse_, attn_sink_, topk_indices_,
                                  topk_length_, dq_, dkv_float, dsink_, dkv_, done_count, tokens_, heads_, head_dim_,
                                  backward_width_, kv_tokens_, selection_width_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnBwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
