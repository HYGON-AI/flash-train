// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/h_post_bda_bwd/fp16_h_post_bda_bwd.hpp"

namespace ftrain {
namespace {

// Resolve fixed launch symbols only while configuring the selected primitive.
template<class Implementation, bool REDUCE_BIAS, bool FP32_MAPPING>
Fp16HPostBdaBwdLaunch bindStreams(int streams) {
    if constexpr (Implementation::kGeneric) {
        return &launchFp16HPostBdaBwd<Implementation, 1, 1, REDUCE_BIAS, FP32_MAPPING>;
    } else {
        constexpr int chunk = Implementation::kChunk;
        switch (streams) {
            case 2:
                if constexpr (chunk <= 2 && 2 % chunk == 0)
                    return &launchFp16HPostBdaBwd<Implementation, 2, chunk, REDUCE_BIAS, FP32_MAPPING>;
                break;
            case 4:
                if constexpr (chunk <= 4 && 4 % chunk == 0)
                    return &launchFp16HPostBdaBwd<Implementation, 4, chunk, REDUCE_BIAS, FP32_MAPPING>;
                break;
            case 8:
                if constexpr (!Implementation::kCoalesced1 && !Implementation::kCoalesced4 && chunk <= 8 &&
                              8 % chunk == 0)
                    return &launchFp16HPostBdaBwd<Implementation, 8, chunk, REDUCE_BIAS, FP32_MAPPING>;
                break;
            case 16:
                if constexpr (!Implementation::kCoalesced1 && !Implementation::kCoalesced4 && chunk <= 8 &&
                              16 % chunk == 0)
                    return &launchFp16HPostBdaBwd<Implementation, 16, chunk, REDUCE_BIAS, FP32_MAPPING>;
                break;
        }
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "configured streams do not support this fixed chunk");
    }
}

template<class Implementation, bool FP32_MAPPING>
Fp16HPostBdaBwdLaunch bindFp16HPostBdaBwd(int streams, bool reduce_bias) {
    return reduce_bias ? bindStreams<Implementation, true, FP32_MAPPING>(streams)
                       : bindStreams<Implementation, false, FP32_MAPPING>(streams);
}

bool hasInt32Indexing(const HPostBdaBwdProblem& problem) noexcept {
    for (const auto* dims :
         {&problem.getGradOutputDims(), &problem.getResidualDims(), &problem.getXDims(), &problem.getBiasDims(),
          &problem.getHResDims(), &problem.getHPostDims(), &problem.getGradResidualDims(), &problem.getGradXDims(),
          &problem.getGradHResDims(), &problem.getGradHPostDims(), &problem.getGradBiasDims()}) {
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

bool supportsShape(const HPostBdaBwdProblem& problem) noexcept {
    return problem.getStreams() > 0 && problem.getStreams() <= 32;
}

}  // namespace

template<class Implementation>
const char* Fp16HPostBdaBwdBase<Implementation>::getName() const noexcept {
    return Implementation::kName;
}

template<class Implementation>
std::unique_ptr<PrimitiveBase> Fp16HPostBdaBwdBase<Implementation>::clone() const {
    return std::make_unique<Implementation>(static_cast<const Implementation&>(*this));
}

template<class Implementation>
Result Fp16HPostBdaBwdBase<Implementation>::isApplicable(const HPostBdaBwdProblem& problem, const Constraints&) const {
    if constexpr (Implementation::kGeneric) {
        const auto streams = problem.getStreams();
        if (streams <= 32 && Implementation::kTileB * streams * (streams + 1) * sizeof(float) > 65536) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "fixed generic tile exceeds the shared-memory budget");
        }
    }
    if constexpr (!Implementation::kGeneric) {
        const auto streams = problem.getStreams();
        if (Implementation::kChunk > (streams == 16 ? 8 : streams) || streams % Implementation::kChunk != 0) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "streams do not support this fixed chunk");
        }
    }
    if constexpr (Implementation::kBiasColumns == 16) {
        if (!problem.hasBiasGradient()) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "bias-column schedule requires a bias-gradient output");
        }
    }
    if (Implementation::kCoalesced1 || Implementation::kCoalesced4) {
        if (!((problem.getStreams() == 2 || problem.getStreams() == 4) && problem.getChannels() >= 512 &&
              problem.getChannels() % (problem.getComputeType() == FTRAIN_NUMERIC_TYPE_FP32 ? 256 : 512) == 0 &&
              ((reinterpret_cast<std::uintptr_t>(problem.getGradOutputAddress()) |
                reinterpret_cast<std::uintptr_t>(problem.getResidualAddress()) |
                reinterpret_cast<std::uintptr_t>(problem.getXAddress())) &
               15) == 0)) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "coalesced path requires aligned vector rows");
        }
    } else {
        if (!(Implementation::kGeneric ||
              ((problem.getStreams() == 2 || problem.getStreams() == 4 || problem.getStreams() == 8 ||
                problem.getStreams() == 16) &&
               ((!Implementation::kVectorized1 && !Implementation::kVectorized4) || problem.getChannels() % 8 == 0)))) {
            return Result(FTRAIN_STATUS_UNSUPPORTED, "algorithm does not support this problem");
        }
    }

    if (problem.getChannels() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this backward implementation requires nonzero channels");
    }

    if (!hasInt32Indexing(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "kernel requires signed 32-bit indexing");
    }
    if (!supportsShape(problem)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "shape exceeds this kernel implementation's tile limits");
    }

    // channel launch rounding adds 255 before division by 256; the expression must fit int.
    if (problem.getChannels() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) - 255) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "channel tile arithmetic exceeds signed 32-bit indexing");
    }
    const std::uint64_t channel_tiles = (problem.getChannels() + 255) / 256;
    // The launcher and fused kernel both use int for batch * channel_tiles + batch.
    if (problem.getBatchStride() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / (channel_tiles + 1)) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "combined backward grid exceeds signed 32-bit indexing");
    }
    // grad_bias remains nonempty for an empty batch; this implementation cannot skip that reduction.
    if (problem.getBatchStride() == 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "this backward implementation requires a nonempty batch");
    }

    const auto mapping_type = problem.getHResNumericType();
    if (mapping_type != FTRAIN_NUMERIC_TYPE_FP16 && mapping_type != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "mapping storage must match activations or use FP32");
    }
    if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_FP16) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "compute type does not match this implementation");
    }
    if (problem.getGradOutputNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getResidualNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32 || problem.getHResNumericType() != mapping_type ||
        problem.getHPostNumericType() != mapping_type ||
        problem.getGradResidualNumericType() != FTRAIN_NUMERIC_TYPE_FP16 ||
        problem.getGradXNumericType() != FTRAIN_NUMERIC_TYPE_FP16 || problem.getGradHResNumericType() != mapping_type ||
        problem.getGradHPostNumericType() != mapping_type ||
        problem.getGradBiasNumericType() != FTRAIN_NUMERIC_TYPE_FP32) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand has an unsupported numeric type");
    }
    if (problem.getGradOutputIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getResidualIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getBiasIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradResidualIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradXIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradHResIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradHPostIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS ||
        problem.getGradBiasIndexType() != FTRAIN_INDEX_TYPE_CONTINUOUS) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand is not continuously indexed");
    }
    const auto mapping_alignment = mapping_type == FTRAIN_NUMERIC_TYPE_FP32 ? alignof(float) : alignof(std::uint16_t);
    if (reinterpret_cast<std::uintptr_t>(problem.getGradOutputAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getResidualAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHResAddress()) % mapping_alignment != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getHPostAddress()) % mapping_alignment != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradResidualAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradXAddress()) % alignof(std::uint16_t) != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHResAddress()) % mapping_alignment != 0 ||
        reinterpret_cast<std::uintptr_t>(problem.getGradHPostAddress()) % mapping_alignment != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an operand address is not aligned for its element type");
    }
    // nullptr omits the separate bias-gradient output and its reduction.
    if (problem.getGradBiasAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getGradBiasAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    // nullptr omits bias; hasBias() disables kernel bias reads.
    if (problem.getBiasAddress() != nullptr &&
        reinterpret_cast<std::uintptr_t>(problem.getBiasAddress()) % alignof(float) != 0) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "an optional operand address is not aligned for its element type");
    }
    if (problem.getGradOutputAddress() == nullptr || problem.getResidualAddress() == nullptr ||
        problem.getXAddress() == nullptr || problem.getHResAddress() == nullptr ||
        problem.getHPostAddress() == nullptr || problem.getGradResidualAddress() == nullptr ||
        problem.getGradXAddress() == nullptr || problem.getGradHResAddress() == nullptr ||
        problem.getGradHPostAddress() == nullptr) {
        return Result(FTRAIN_STATUS_UNSUPPORTED, "a required operand address is null");
    }
    return Result{};
}

template<class Implementation>
std::uint64_t Fp16HPostBdaBwdBase<Implementation>::getRequiredWorkspaceBytes() const noexcept {
    return 0;
}

template<class Implementation>
void Fp16HPostBdaBwdBase<Implementation>::configure(const HPostBdaBwdProblem& problem) {
    grad_output_   = problem.getGradOutputAddress();
    residual_      = problem.getResidualAddress();
    x_             = problem.getXAddress();
    bias_          = static_cast<const float*>(problem.getBiasAddress());
    h_res_         = problem.getHResAddress();
    h_post_        = problem.getHPostAddress();
    grad_residual_ = problem.getGradResidualAddress();
    grad_x_        = problem.getGradXAddress();
    grad_h_res_    = problem.getGradHResAddress();
    grad_h_post_   = problem.getGradHPostAddress();
    grad_bias_     = static_cast<float*>(problem.getGradBiasAddress());
    batch_stride_  = static_cast<int>(problem.getBatchStride());
    num_streams_   = static_cast<int>(problem.getStreams());
    channels_      = static_cast<int>(problem.getChannels());
    has_bias_      = problem.hasBias();
    reduce_bias_   = problem.hasBiasGradient();
    launch_        = problem.getHResNumericType() == FTRAIN_NUMERIC_TYPE_FP32
                         ? bindFp16HPostBdaBwd<Implementation, true>(num_streams_, reduce_bias_)
                         : bindFp16HPostBdaBwd<Implementation, false>(num_streams_, reduce_bias_);
}

template<class Implementation>
void Fp16HPostBdaBwdBase<Implementation>::executeImpl(const Resources& resources) {
    const bool launched =
        launch_(resources.getStream(), grad_output_, residual_, x_, bias_, h_res_, h_post_, grad_residual_, grad_x_,
                grad_h_res_, grad_h_post_, reduce_bias_ ? grad_bias_ : nullptr, batch_stride_, num_streams_, channels_,
                has_bias_);
    if (!launched) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR,
                        "HPostBdaBwd kernel rejected a shape its applicability check accepted");
    }
}

template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced1>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwd>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdBias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1Chunk2>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1Chunk2Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1Chunk4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1Chunk4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1Chunk8>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled1Chunk8Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4Chunk2>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4Chunk2Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4Chunk4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4Chunk4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4Chunk8>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTiled4Chunk8Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1Chunk2>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1Chunk2Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1Chunk4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1Chunk4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1Chunk8>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized1Chunk8Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4Chunk2>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4Chunk2Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4Chunk4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4Chunk4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4Chunk8>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdVectorized4Chunk8Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced1Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced1Chunk2>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced1Chunk2Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced1Chunk4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced1Chunk4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced4Chunk2>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced4Chunk2Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced4Chunk4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdCoalesced4Chunk4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB1>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB1Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB2>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB2Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB4>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB4Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB8>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB8Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB16Bias16>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB32>;
template class Fp16HPostBdaBwdBase<Fp16HPostBdaBwdTileB32Bias16>;

}  // namespace ftrain
