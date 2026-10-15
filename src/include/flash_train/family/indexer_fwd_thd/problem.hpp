// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_FWD_THD_PROBLEM_HPP_
#define FTRAIN_FAMILY_INDEXER_FWD_THD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/indexer_fwd_thd.hpp"

namespace ftrain {

// One IndexerFwdThd call's parameters as a support checklist: the packed
// (THD) indexer forward, whose per-segment causal bounds come from
// cu_seqlens_q / cu_seqlens_kv instead of a leading batch dimension.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class IndexerFwdThdProblem final {
  public:
    IndexerFwdThdProblem(Tensor q, Tensor k, Tensor w, Tensor cu_seqlens_q, Tensor cu_seqlens_kv,
                         Tensor q_causal_offsets, Tensor scores, Tensor ratio, Tensor sm_scale, Tensor max_seqlen_kv,
                         IndexerFwdThdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKDims() const noexcept { return k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWDims() const noexcept { return w_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getCuSeqlensQDims() const noexcept {
        return cu_seqlens_q_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getCuSeqlensKvDims() const noexcept {
        return cu_seqlens_kv_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getQCausalOffsetsDims() const noexcept {
        return q_causal_offsets_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getScoresDims() const noexcept { return scores_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSmScaleDims() const noexcept { return sm_scale_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getMaxSeqlenKvDims() const noexcept {
        return max_seqlen_kv_.getStorageView().getDims();
    }

    std::uint64_t getTotalQ() const noexcept { return static_cast<std::uint64_t>(q_.getStorageView().getDims()[0]); }

    std::uint64_t getTotalK() const noexcept { return static_cast<std::uint64_t>(k_.getStorageView().getDims()[0]); }

    std::uint64_t getMaxSeqlenKv() const noexcept { return static_cast<std::uint64_t>(max_seqlen_kv_value_); }

    std::uint64_t getNumSequences() const noexcept {
        return static_cast<std::uint64_t>(cu_seqlens_q_.getStorageView().getDims()[0]) - 1;
    }

    std::uint64_t getQueryHeads() const noexcept {
        return static_cast<std::uint64_t>(q_.getStorageView().getDims()[1]);
    }

    std::uint64_t getHeadDim() const noexcept { return static_cast<std::uint64_t>(q_.getStorageView().getDims()[2]); }

    // --- attributes ---

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getRatio() const noexcept { return ratio_value_; }

    // --- element types ---

    FTrainNumericType getQNumericType() const noexcept { return q_.getStorageView().getNumericType(); }
    FTrainNumericType getKNumericType() const noexcept { return k_.getStorageView().getNumericType(); }
    FTrainNumericType getWNumericType() const noexcept { return w_.getStorageView().getNumericType(); }
    FTrainNumericType getScoresNumericType() const noexcept { return scores_.getStorageView().getNumericType(); }
    FTrainNumericType getCuSeqLensQNumericType() const noexcept {
        return cu_seqlens_q_.getStorageView().getNumericType();
    }
    FTrainNumericType getCuSeqLensKvNumericType() const noexcept {
        return cu_seqlens_kv_.getStorageView().getNumericType();
    }
    FTrainNumericType getCausalOffsetsNumericType() const noexcept {
        return q_causal_offsets_.getStorageView().getNumericType();
    }

    // --- layouts ---

    FTrainIndexType getQIndexType() const noexcept { return q_.getStorageView().getIndexType(); }
    FTrainIndexType getKIndexType() const noexcept { return k_.getStorageView().getIndexType(); }
    FTrainIndexType getWIndexType() const noexcept { return w_.getStorageView().getIndexType(); }
    FTrainIndexType getScoresIndexType() const noexcept { return scores_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensQIndexType() const noexcept { return cu_seqlens_q_.getStorageView().getIndexType(); }
    FTrainIndexType getCuSeqLensKvIndexType() const noexcept { return cu_seqlens_kv_.getStorageView().getIndexType(); }
    FTrainIndexType getCausalOffsetsIndexType() const noexcept {
        return q_causal_offsets_.getStorageView().getIndexType();
    }

    // --- addresses; q_causal_offsets is optional and may be null ---

    void* getQAddress() const noexcept { return q_.getStorageView().getMemory(); }
    void* getKAddress() const noexcept { return k_.getStorageView().getMemory(); }
    void* getWAddress() const noexcept { return w_.getStorageView().getMemory(); }
    void* getScoresAddress() const noexcept { return scores_.getStorageView().getMemory(); }
    void* getCuSeqLensQAddress() const noexcept { return cu_seqlens_q_.getStorageView().getMemory(); }
    void* getCuSeqLensKvAddress() const noexcept { return cu_seqlens_kv_.getStorageView().getMemory(); }
    void* getCausalOffsetsAddress() const noexcept { return q_causal_offsets_.getStorageView().getMemory(); }

    // --- selection cache ---

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }
    FTrainNumericType getSmScaleNumericType() const noexcept { return sm_scale_.getStorageView().getNumericType(); }
    FTrainIndexType getSmScaleIndexType() const noexcept { return sm_scale_.getStorageView().getIndexType(); }
    void* getSmScaleAddress() const noexcept { return sm_scale_.getStorageView().getMemory(); }
    FTrainNumericType getMaxSeqlenKvNumericType() const noexcept {
        return max_seqlen_kv_.getStorageView().getNumericType();
    }
    FTrainIndexType getMaxSeqlenKvIndexType() const noexcept { return max_seqlen_kv_.getStorageView().getIndexType(); }
    void* getMaxSeqlenKvAddress() const noexcept { return max_seqlen_kv_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    Tensor q_;
    Tensor k_;
    Tensor w_;
    Tensor cu_seqlens_q_;
    Tensor cu_seqlens_kv_;
    Tensor q_causal_offsets_;
    Tensor scores_;
    Tensor ratio_;
    Tensor sm_scale_;
    Tensor max_seqlen_kv_;
    std::int64_t ratio_value_{};
    std::int64_t max_seqlen_kv_value_{};
    IndexerFwdThdAttributes attributes_;
};

}  // namespace ftrain

#endif
