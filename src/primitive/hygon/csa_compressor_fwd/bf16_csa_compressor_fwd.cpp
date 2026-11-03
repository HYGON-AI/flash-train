// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/csa_compressor_fwd/bf16_csa_compressor_fwd.hpp"

namespace ftrain {
namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation>
Bf16CSACompressorFwdLaunch bindBf16CSACompressorFwd(int coff) {
    constexpr int capacity = Implementation::kCapacity;
    constexpr int dim      = Implementation::kDim;
    return coff == 2 ? &launchBf16CSACompressorFwd<Implementation, capacity, dim, true>
                     : &launchBf16CSACompressorFwd<Implementation, capacity, dim, false>;
}

bool hasInt32Indexing(const CSACompressorFwdProblem& problem) noexcept {
    if (problem.getRatio() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getHeadDim() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getCoff() > std::numeric_limits<std::int32_t>::max()) { return false; }
    if (problem.getTotalComp() > std::numeric_limits<std::int32_t>::max()) { return false; }
    for (const auto* dims :
         {&problem.getKvDims(), &problem.getScoreDims(), &problem.getApeDims(), &problem.getCuSeqlensDims(),
          &problem.getCuSeqlensCompDims(), &problem.getCompressedDims(), &problem.getRatioDims(),
          &problem.getHeadDimDims(), &problem.getCoffDims(), &problem.getTotalCompDims()}) {
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

bool supportsParameters(const CSACompressorFwdProblem& problem) noexcept {
    return (problem.getRatio() == 4 || problem.getRatio() == 128) && (problem.getCoff() == 1 || problem.getCoff() == 2);
}

}  // namespace

template<class Implementation>
const char* Bf16CSACompressorFwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Bf16CSACompressorFwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Bf16CSACompressorFwdBase<Implementation>::isApplicable(const CSACompressorFwdProblem& problem,
                                                              const Constraints&) const {
    if (!(Implementation::kSerial ||
          (Implementation::kCached && problem.getRatio() * problem.getCoff() <= Implementation::kCapacity) ||
          (Implementation::kTiled && problem.getRatio() * problem.getCoff() <= Implementation::kCapacity) ||
          ((Implementation::kR128 || Implementation::kR128WindowMajor) && problem.getRatio() == 128 &&
           (problem.getHeadDim() == 128 || problem.getHeadDim() == 512)))) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
    }

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
        problem.getCompressedNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
        problem.getRatioNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getHeadDimNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getCoffNumericType() != FTRAIN_NUMERIC_TYPE_INT64 ||
        problem.getTotalCompNumericType() != FTRAIN_NUMERIC_TYPE_INT64) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getKvIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getScoreIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getApeIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCuSeqLensCompIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCompressedIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getRatioIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHeadDimIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getCoffIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getTotalCompIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    if (reinterpret_cast<std::uintptr_t>(problem.getKvAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getScoreAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getApeAddress()) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCuSeqLensCompAddress()) % alignof(std::int32_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCompressedAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getRatioAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHeadDimAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getCoffAddress()) % alignof(std::int64_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getTotalCompAddress()) % alignof(std::int64_t) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    if (problem.getKvAddress() == nullptr || problem.getScoreAddress() == nullptr ||
        problem.getApeAddress() == nullptr || problem.getCuSeqLensAddress() == nullptr ||
        problem.getCuSeqLensCompAddress() == nullptr || problem.getRatioAddress() == nullptr ||
        problem.getCoffAddress() == nullptr || problem.getHeadDimAddress() == nullptr ||
        problem.getTotalCompAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    if (!supportsParameters(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "ratio must be 4 or 128 and coff must be 1 or 2");
    }
    // A zero-row request writes nothing; the caller reads the empty output
    // rather than a launch, matching the operator's upstream contract.
    if (problem.getTotalComp() == 0) { return Result{}; }
    if (problem.getCompressedAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Bf16CSACompressorFwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Bf16CSACompressorFwdBase<Implementation>::configure(const CSACompressorFwdProblem& problem) {
    kv_              = problem.getKvAddress();
    score_           = problem.getScoreAddress();
    ape_             = static_cast<const float*>(problem.getApeAddress());
    cu_seqlens_      = static_cast<const int*>(problem.getCuSeqLensAddress());
    cu_seqlens_comp_ = static_cast<const int*>(problem.getCuSeqLensCompAddress());
    compressed_      = problem.getCompressedAddress();
    total_comp_      = static_cast<int>(problem.getTotalComp());
    num_sequences_   = static_cast<int>(problem.getNumSequences());
    ratio_           = static_cast<int>(problem.getRatio());
    head_dim_        = static_cast<int>(problem.getHeadDim());
    coff_            = static_cast<int>(problem.getCoff());
    launch_          = bindBf16CSACompressorFwd<Implementation>(coff_);
    total_tokens_    = static_cast<int>(problem.getTotalTokens());
}

template<class Implementation>
void Bf16CSACompressorFwdBase<Implementation>::executeImpl(const Resources& resources) {
    if (total_comp_ == 0) { return; }
    const bool launched = launch_(resources.getStream(), kv_, score_, ape_, cu_seqlens_, cu_seqlens_comp_, compressed_,
                                  total_comp_, num_sequences_, ratio_, head_dim_, coff_, total_tokens_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "CSACompressorFwd kernel rejected a shape its applicability check accepted");
    }
}

template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwd>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCached>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiled>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdR128>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdR128WindowMajor>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity8>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity16>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity32>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdCachedCapacity64>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity128>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity160>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256Dim16>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256Dim8>;
template class Bf16CSACompressorFwdBase<Bf16CSACompressorFwdTiledCapacity256Dim4>;

}  // namespace ftrain
