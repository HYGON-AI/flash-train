// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/dense_indexer_score/fp32_dense_indexer_score.hpp"

namespace ftrain {
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

const char* Fp32DenseIndexerScore::getName() const noexcept { return "Fp32DenseIndexerScore"; }

std::unique_ptr<PrimitiveBase> Fp32DenseIndexerScore::clone() const {
    return std::make_unique<Fp32DenseIndexerScore>(*this);
}

Result Fp32DenseIndexerScore::isApplicable(const DenseIndexerScoreProblem& problem, const Constraints&) const {
    if (problem.getBatch() == 0 || problem.getQueryHeads() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this implementation requires nonempty launch dimensions");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getQNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getKNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getWNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
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
    if (reinterpret_cast<std::uintptr_t>(problem.getQAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getKAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getWAddress()) % alignof(float) != 0 ||
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
    if (!fp32DenseIndexerScoreDenseMmacSupported()) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "the device does not expose the MMAC path this kernel needs");
    }
    if (!fp32DenseIndexerScoreLaunchFits(static_cast<int>(problem.getQueryHeads()),
                                         static_cast<int>(problem.getHeadDim()),
                                         static_cast<int>(problem.getKeyLength()))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds the instantiated kernel or shared-memory coverage");
    }
    return Result{};
}

std::uint64_t Fp32DenseIndexerScore::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Fp32DenseIndexerScore::configure(const DenseIndexerScoreProblem& problem) {
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
}

void Fp32DenseIndexerScore::executeImpl(const Resources& resources) {
    const bool launched =
        launchFp32DenseIndexerScore(resources.getStream(), q_, k_, w_, out_, denom_, batch_, query_length_, key_length_,
                                    query_heads_, /*kv_heads=*/1, head_dim_, ratio_, sm_scale_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "DenseIndexerScore kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
