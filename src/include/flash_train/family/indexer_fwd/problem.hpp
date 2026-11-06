// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_FWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_INDEXER_FWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/indexer_fwd.hpp"

namespace ftrain {

// One IndexerFwd call's parameters as a support checklist: walking the
// getters below is the whole list of properties an implementation must
// decide whether it supports. The constructor validates ranks, one KV head,
// cross-port shapes, byte overflow and memory placement. Unsupported storage
// types, layouts and kernel index widths are selection decisions.
//
// The operator computes, for every query row,
// scores[b,q,k] = sm_scale * sum_h ReLU(q[b,q,h,:] . k[b,k,0,:]) * w[b,q,h]
// under the ratio-causal mask the kernel applies.
class IndexerFwdProblem final {
  public:
    // Takes the four data ports and two host scalar ports by value with the call's
    // attributes. See the class comment for the validation this performs;
    // allocation failure throws std::bad_alloc.
    IndexerFwdProblem(Tensor q, Tensor k, Tensor w, Tensor scores, Tensor ratio, Tensor sm_scale,
                      IndexerFwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKDims() const noexcept { return k_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getWDims() const noexcept { return w_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getScoresDims() const noexcept { return scores_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getRatioDims() const noexcept { return ratio_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSmScaleDims() const noexcept { return sm_scale_.getStorageView().getDims(); }

    std::uint64_t getBatch() const noexcept;
    std::uint64_t getQueryLength() const noexcept;
    std::uint64_t getKeyLength() const noexcept;
    std::uint64_t getQueryHeads() const noexcept;
    std::uint64_t getHeadDim() const noexcept;

    // --- attributes ---

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }
    std::int64_t getRatio() const noexcept { return ratio_value_; }

    // --- element types ---

    FTrainNumericType getQNumericType() const noexcept;
    FTrainNumericType getKNumericType() const noexcept;
    FTrainNumericType getWNumericType() const noexcept;
    FTrainNumericType getScoresNumericType() const noexcept;

    // --- layouts ---

    FTrainIndexType getQIndexType() const noexcept;
    FTrainIndexType getKIndexType() const noexcept;
    FTrainIndexType getWIndexType() const noexcept;
    FTrainIndexType getScoresIndexType() const noexcept;

    // --- addresses; every port is required for this operator ---

    void* getQAddress() const noexcept;
    void* getKAddress() const noexcept;
    void* getWAddress() const noexcept;
    void* getScoresAddress() const noexcept;

    // --- selection cache ---

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getRatioNumericType() const noexcept { return ratio_.getStorageView().getNumericType(); }
    FTrainIndexType getRatioIndexType() const noexcept { return ratio_.getStorageView().getIndexType(); }
    void* getRatioAddress() const noexcept { return ratio_.getStorageView().getMemory(); }
    FTrainNumericType getSmScaleNumericType() const noexcept { return sm_scale_.getStorageView().getNumericType(); }
    FTrainIndexType getSmScaleIndexType() const noexcept { return sm_scale_.getStorageView().getIndexType(); }
    void* getSmScaleAddress() const noexcept { return sm_scale_.getStorageView().getMemory(); }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    Tensor q_;
    Tensor k_;
    Tensor w_;
    Tensor scores_;
    Tensor ratio_;
    Tensor sm_scale_;
    std::int64_t ratio_value_{};
    IndexerFwdAttributes attributes_;
};

}  // namespace ftrain

#endif
