// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/sparse_attn_fwd/fp32_sparse_attn_fwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const SparseAttnFwdProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getQDims(), &problem.getKvDims(), &problem.getAttnSinkDims(), &problem.getTopkIndicesDims(),
          &problem.getTopkLengthDims(), &problem.getOutDims(), &problem.getLseDims(), &problem.getSoftmaxScaleDims()}) {
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

bool supportsValueWidth(const SparseAttnFwdProblem& problem) noexcept { return problem.getValueDim() == 512; }

// The fused forward stages Q/K head rows up to this width, and the value
// width is fixed by the kernel's d_v specialization.
constexpr std::uint64_t kMaxHeadDim = 576;
constexpr std::uint64_t kValueDim   = 512;

}  // namespace

const char* Fp32SparseAttnFwd::getName() const noexcept { return "Fp32SparseAttnFwd"; }

std::unique_ptr<PrimitiveBase> Fp32SparseAttnFwd::clone() const { return std::make_unique<Fp32SparseAttnFwd>(*this); }

Result Fp32SparseAttnFwd::isApplicable(const SparseAttnFwdProblem& problem, const Constraints&) const {
    if (problem.getHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getAttnSinkNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getTopkIndicesNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getTopkLengthNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getOutNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getLseNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getSoftmaxScaleNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getQIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getAttnSinkIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkIndicesIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTopkLengthIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getOutIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getLseIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getSoftmaxScaleIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getAttnSinkAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkIndicesAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTopkLengthAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getOutAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getLseAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getSoftmaxScaleAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getQAddress() == nullptr || problem.getKvAddress() == nullptr ||
        problem.getAttnSinkAddress() == nullptr || problem.getTopkIndicesAddress() == nullptr ||
        problem.getOutAddress() == nullptr || problem.getLseAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!problem.hasNonEmptyInputs()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the request has no kernel work"); }
    if (!problem.hasSingleKvHead()) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the kernel supports one KV head"); }
    if (problem.getHeadDim() < kValueDim || problem.getHeadDim() > kMaxHeadDim || problem.getHeadDim() % 16 != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "Q/K head dimension is outside the kernel's staged width");
    }
    if (!supportsValueWidth(problem)) { return Result(FTRAIN_STATUS_UNSUPPORTED, "the value width must be 512"); }
    if (!problem.hasNonEmptySelection()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "selection width must be positive");
    }
    if (!fp32SparseAttnFwdSparseAttnFusedSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED,
                      "the device does not expose the MMAC sparse-attention path this kernel needs");
    }
    if (!fp32SparseAttnFwdLaunchFits(static_cast<int>(problem.getHeadDim()),
                                     static_cast<int>(problem.getSelectionWidth()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Fp32SparseAttnFwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp32SparseAttnFwd::configure(const SparseAttnFwdProblem& problem) {
    q_               = problem.getQAddress();
    kv_              = problem.getKvAddress();
    attn_sink_       = static_cast<const float*>(problem.getAttnSinkAddress());
    topk_indices_    = static_cast<const int*>(problem.getTopkIndicesAddress());
    topk_length_     = static_cast<const int*>(problem.getTopkLengthAddress());
    out_             = problem.getOutAddress();
    lse_             = static_cast<float*>(problem.getLseAddress());
    tokens_          = static_cast<int>(problem.getTokens());
    heads_           = static_cast<int>(problem.getHeads());
    head_dim_        = static_cast<int>(problem.getHeadDim());
    value_dim_       = static_cast<int>(problem.getValueDim());
    kv_tokens_       = static_cast<int>(problem.getKvTokens());
    selection_width_ = static_cast<int>(problem.getSelectionWidth());
    softmax_scale_   = *static_cast<const float*>(problem.getSoftmaxScaleAddress());
}

void Fp32SparseAttnFwd::executeImpl(const Resources& resources) {
    const bool launched =
        launchFp32SparseAttnFwd(resources.getStream(), q_, kv_, attn_sink_, topk_indices_, out_, lse_, topk_length_,
                                tokens_, heads_, head_dim_, value_dim_, kv_tokens_, selection_width_, softmax_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "SparseAttnFwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
