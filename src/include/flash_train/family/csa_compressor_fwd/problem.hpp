// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/csa_compressor_fwd.hpp"

namespace ftrain {

// One CSACompressorFwd call's parameters as a support checklist: the fused
// gated pooling that turns a packed token stream into compressed blocks.
//
// This family has no dtype chain: the kernel reads BF16 key/score rows and an
// FP32 position table and writes BF16 blocks. The constructor validates the
// packed shapes against the attributes (head_dim, coff, ratio, total_comp),
// the cumulative-length metadata, and 32-bit-indexable operands.
class CSACompressorFwdProblem final {
  public:
    CSACompressorFwdProblem(Tensor kv, Tensor score, Tensor ape, Tensor cu_seqlens, Tensor cu_seqlens_comp,
                            Tensor compressed, Tensor ratio, Tensor head_dim, Tensor coff, Tensor total_comp,
                            CSACompressorFwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getKvDims() const noexcept { return kv_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getScoreDims() const noexcept { return score_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getApeDims() const noexcept { return ape_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getCuSeqlensDims() const noexcept {
        return cu_seqlens_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getCuSeqlensCompDims() const noexcept {
        return cu_seqlens_comp_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getCompressedDims() const noexcept {
        return compressed_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getHeadDimDims() const noexcept { return head_dim_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getCoffDims() const noexcept { return coff_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTotalCompDims() const noexcept {
        return total_comp_.getStorageView().getDims();
    }

    std::uint64_t getTotalTokens() const noexcept { return dim(kv_, 0); }
    std::uint64_t getWidth() const noexcept { return dim(kv_, 1); }
    std::uint64_t getNumSequences() const noexcept { return dim(cu_seqlens_, 0) - 1; }
    std::uint64_t getTotalCompressed() const noexcept { return dim(compressed_, 0); }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    std::int64_t getRatio() const noexcept { return ratio_value_; }
    std::int64_t getHeadDim() const noexcept { return head_dim_value_; }
    std::int64_t getCoff() const noexcept { return coff_value_; }
    std::int64_t getTotalComp() const noexcept { return total_comp_value_; }

    FTrainNumericType getKvNumericType() const noexcept { return kv_.getStorageView().getNumericType(); }
    FTrainNumericType getScoreNumericType() const noexcept { return score_.getStorageView().getNumericType(); }
    FTrainNumericType getApeNumericType() const noexcept { return ape_.getStorageView().getNumericType(); }
    FTrainNumericType getCuSeqLensNumericType() const noexcept { return cu_seqlens_.getStorageView().getNumericType(); }
    FTrainNumericType getCuSeqLensCompNumericType() const noexcept {
        return cu_seqlens_comp_.getStorageView().getNumericType();
    }
    FTrainNumericType getCompressedNumericType() const noexcept {
        return compressed_.getStorageView().getNumericType();
    }

    FTrainIndexType getKvIndexType() const noexcept { return kv_.getStorageView().getIndexType(); }
    FTrainIndexType getScoreIndexType() const noexcept { return score_.getStorageView().getIndexType(); }
    FTrainIndexType getApeIndexType() const noexcept { return ape_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensIndexType() const noexcept { return cu_seqlens_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensCompIndexType() const noexcept {
        return cu_seqlens_comp_.getStorageView().getIndexType();
    }
    FTrainIndexType getCompressedIndexType() const noexcept { return compressed_.getStorageView().getIndexType(); }

    // getCompressedAddress may be null when total_comp is zero.
    void* getKvAddress() const noexcept { return kv_.getStorageView().getMemory(); }
    void* getScoreAddress() const noexcept { return score_.getStorageView().getMemory(); }
    void* getApeAddress() const noexcept { return ape_.getStorageView().getMemory(); }
    void* getCuSeqLensAddress() const noexcept { return cu_seqlens_.getStorageView().getMemory(); }
    void* getCuSeqLensCompAddress() const noexcept { return cu_seqlens_comp_.getStorageView().getMemory(); }
    void* getCompressedAddress() const noexcept { return compressed_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }
    FTrainNumericType getHeadDimNumericType() const noexcept { return head_dim_.getStorageView().getNumericType(); }
    FTrainIndexType getHeadDimIndexType() const noexcept { return head_dim_.getStorageView().getIndexType(); }
    void* getHeadDimAddress() const noexcept { return head_dim_.getStorageView().getMemory(); }
    FTrainNumericType getCoffNumericType() const noexcept { return coff_.getStorageView().getNumericType(); }
    FTrainIndexType getCoffIndexType() const noexcept { return coff_.getStorageView().getIndexType(); }
    void* getCoffAddress() const noexcept { return coff_.getStorageView().getMemory(); }
    FTrainNumericType getTotalCompNumericType() const noexcept { return total_comp_.getStorageView().getNumericType(); }
    FTrainIndexType getTotalCompIndexType() const noexcept { return total_comp_.getStorageView().getIndexType(); }
    void* getTotalCompAddress() const noexcept { return total_comp_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor kv_;
    Tensor score_;
    Tensor ape_;
    Tensor cu_seqlens_;
    Tensor cu_seqlens_comp_;
    Tensor compressed_;
    Tensor ratio_;
    Tensor head_dim_;
    Tensor coff_;
    Tensor total_comp_;
    std::int64_t ratio_value_{};
    std::int64_t head_dim_value_{};
    std::int64_t coff_value_{};
    std::int64_t total_comp_value_{};
    CSACompressorFwdAttributes attributes_;
};

}  // namespace ftrain

#endif
