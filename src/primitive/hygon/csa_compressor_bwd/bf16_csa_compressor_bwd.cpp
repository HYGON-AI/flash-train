// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/csa_compressor_bwd/bf16_csa_compressor_bwd.hpp"

namespace ftrain {
namespace {

bool hasInt32Indexing(const CSACompressorBwdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getHeadDim() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getCoff() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims :
         {&problem.getKvDims(), &problem.getScoreDims(), &problem.getApeDims(), &problem.getCuSeqlensDims(),
          &problem.getCuSeqlensCompDims(), &problem.getGradCompressedDims(), &problem.getGradKvDims(),
          &problem.getGradScoreDims(), &problem.getGradApeDims(), &problem.getRatioDims(), &problem.getHeadDimDims(),
          &problem.getCoffDims()}) {
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

bool supportsParameters(const CSACompressorBwdProblem& problem) noexcept {
    return (problem.getRatio() == 4 || problem.getRatio() == 128) && (problem.getCoff() == 1 || problem.getCoff() == 2);
}

// Byte sizes of the accumulator outputs the launch owns.
std::uint64_t byteSize(std::uint64_t elements, std::uint64_t element_bytes) noexcept {
    return elements * element_bytes;
}

}  // namespace

const char* Bf16CSACompressorBwd::getName() const noexcept { return "CSACompressorBwd"; }

std::unique_ptr<PrimitiveBase> Bf16CSACompressorBwd::clone() const {
    return std::make_unique<Bf16CSACompressorBwd>(*this);
}

Result Bf16CSACompressorBwd::isApplicable(const CSACompressorBwdProblem& problem, const Constraints&) const {
    // Padding rows in this kernel gather the first ratio tokens.
    if (problem.getTotalTokens() < static_cast<std::uint64_t>(problem.getRatio())) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "padding gather requires at least ratio tokens");
    }

    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type is not bf16");
    }
    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }

    if (problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getScoreNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getApeNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getCuSeqLensNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getCuSeqLensCompNumericType() != FTRAIN_NUMERIC_TYPE_INT32 ||
        problem.getGradCompressedNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getGradKvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getGradScoreNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getGradApeNumericType() != FTRAIN_NUMERIC_TYPE_FP32 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getHeadDimNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getCoffNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getApeIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensCompIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradCompressedIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradApeIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHeadDimIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCoffIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getScoreAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getApeAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensCompAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradCompressedAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradKvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradScoreAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradApeAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHeadDimAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCoffAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getKvAddress() == nullptr || problem.getScoreAddress() == nullptr ||
        problem.getApeAddress() == nullptr || problem.getCuSeqLensAddress() == nullptr ||
        problem.getCuSeqLensCompAddress() == nullptr || problem.getGradKvAddress() == nullptr ||
        problem.getGradScoreAddress() == nullptr || problem.getGradApeAddress() == nullptr ||
        problem.getRatioAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!supportsParameters(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "ratio must be 4 or 128 and coff must be 1 or 2");
    }
    // A zero-row request still zeroes the accumulators but launches nothing.
    if (problem.getTotalCompressed() != 0 && problem.getGradCompressedAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

std::uint64_t Bf16CSACompressorBwd::getRequiredWorkspaceBytes() const noexcept { return 0; }

void Bf16CSACompressorBwd::configure(const CSACompressorBwdProblem& problem) {
    kv_                       = problem.getKvAddress();
    score_                    = problem.getScoreAddress();
    ape_                      = static_cast<const float*>(problem.getApeAddress());
    cu_seqlens_               = static_cast<const int*>(problem.getCuSeqLensAddress());
    cu_seqlens_comp_          = static_cast<const int*>(problem.getCuSeqLensCompAddress());
    grad_compressed_          = problem.getGradCompressedAddress();
    grad_kv_                  = problem.getGradKvAddress();
    grad_score_               = problem.getGradScoreAddress();
    grad_ape_                 = static_cast<float*>(problem.getGradApeAddress());
    total_comp_               = static_cast<int>(problem.getTotalCompressed());
    num_sequences_            = static_cast<int>(problem.getNumSequences());
    ratio_                    = static_cast<int>(problem.getRatio());
    head_dim_                 = static_cast<int>(problem.getHeadDim());
    coff_                     = static_cast<int>(problem.getCoff());
    total_tokens_             = static_cast<int>(problem.getTotalTokens());
    const std::uint64_t width = static_cast<std::uint64_t>(problem.getWidth());
    grad_kv_bytes_            = byteSize(static_cast<std::uint64_t>(total_tokens_) * width, 2);
    grad_score_bytes_         = grad_kv_bytes_;
    grad_ape_bytes_           = byteSize(static_cast<std::uint64_t>(problem.getRatio()) * width, sizeof(float));
}

void Bf16CSACompressorBwd::executeImpl(const Resources& resources) {
    // The accumulators start at exact zero: never-consumed positions must read
    // back as zero, and the atomic additions must not add into stale data.
    bf16CSACompressorBwdZeroWorkspace(resources.getStream(), grad_kv_, grad_kv_bytes_);
    bf16CSACompressorBwdZeroWorkspace(resources.getStream(), grad_score_, grad_score_bytes_);
    bf16CSACompressorBwdZeroWorkspace(resources.getStream(), grad_ape_, grad_ape_bytes_);
    if (total_comp_ == 0) { return; }
    const bool launched = launchBf16CSACompressorBwd(
        resources.getStream(), kv_, score_, ape_, cu_seqlens_, cu_seqlens_comp_, grad_compressed_, grad_kv_,
        grad_score_, grad_ape_, total_comp_, num_sequences_, ratio_, head_dim_, coff_, total_tokens_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "CSACompressorBwd kernel rejected a shape its applicability check accepted");
    }
}

}  // namespace ftrain
