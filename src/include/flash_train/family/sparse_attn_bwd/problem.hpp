// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_BWD_PROBLEM_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_BWD_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"
#include "flash_train/operation/sparse_attn_bwd.hpp"

namespace ftrain {

// One SparseAttnBwd call's parameters as a support checklist: the sparse
// attention backward, returning dq, dkv, and the attention-sink gradient.
//
// The constructor enforces ranks, cross-port shapes, scalar ABI and memory
// placement. Getters expose problem properties; each Primitive decides
// which values its implementation supports.
class SparseAttnBwdProblem final {
  public:
    SparseAttnBwdProblem(Tensor q, Tensor kv, Tensor out, Tensor dout, Tensor lse, Tensor attn_sink,
                         Tensor topk_indices, Tensor topk_length, Tensor dq, Tensor dkv, Tensor dsink,
                         Tensor softmax_scale, SparseAttnBwdAttributes attributes);

    // --- shape ---

    const std::vector<std::int64_t>& getQDims() const noexcept { return q_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getKvDims() const noexcept { return kv_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getOutDims() const noexcept { return out_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDoutDims() const noexcept { return dout_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getLseDims() const noexcept { return lse_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getAttnSinkDims() const noexcept { return attn_sink_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getTopkIndicesDims() const noexcept {
        return topk_indices_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getTopkLengthDims() const noexcept {
        return topk_length_.getStorageView().getDims();
    }
    const std::vector<std::int64_t>& getDqDims() const noexcept { return dq_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDkvDims() const noexcept { return dkv_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getDsinkDims() const noexcept { return dsink_.getStorageView().getDims(); }
    const std::vector<std::int64_t>& getSoftmaxScaleDims() const noexcept {
        return softmax_scale_.getStorageView().getDims();
    }

    std::uint64_t getTokens() const noexcept { return dim(q_, 0); }
    std::uint64_t getKvTokens() const noexcept { return dim(kv_, 0); }
    std::uint64_t getHeads() const noexcept { return dim(q_, 1); }
    std::uint64_t getHeadDim() const noexcept { return dim(q_, 2); }
    std::uint64_t getValueDim() const noexcept { return dim(out_, 2); }
    std::uint64_t getSelectionWidth() const noexcept { return dim(topk_indices_, 1); }

    FTrainNumericType getComputeType() const noexcept { return attributes_.getComputeType(); }

    FTrainNumericType getQNumericType() const noexcept { return q_.getStorageView().getNumericType(); }
    FTrainNumericType getKvNumericType() const noexcept { return kv_.getStorageView().getNumericType(); }
    FTrainNumericType getOutNumericType() const noexcept { return out_.getStorageView().getNumericType(); }
    FTrainNumericType getDoutNumericType() const noexcept { return dout_.getStorageView().getNumericType(); }
    FTrainNumericType getLseNumericType() const noexcept { return lse_.getStorageView().getNumericType(); }
    FTrainNumericType getAttnSinkNumericType() const noexcept { return attn_sink_.getStorageView().getNumericType(); }
    FTrainNumericType getTopkIndicesNumericType() const noexcept {
        return topk_indices_.getStorageView().getNumericType();
    }
    FTrainNumericType getTopkLengthNumericType() const noexcept {
        return topk_length_.getStorageView().getNumericType();
    }
    FTrainNumericType getDqNumericType() const noexcept { return dq_.getStorageView().getNumericType(); }
    FTrainNumericType getDkvNumericType() const noexcept { return dkv_.getStorageView().getNumericType(); }
    FTrainNumericType getDsinkNumericType() const noexcept { return dsink_.getStorageView().getNumericType(); }

    FTrainIndexType getQIndexType() const noexcept { return q_.getStorageView().getIndexType(); }
    FTrainIndexType getKvIndexType() const noexcept { return kv_.getStorageView().getIndexType(); }
    FTrainIndexType getOutIndexType() const noexcept { return out_.getStorageView().getIndexType(); }
    FTrainIndexType getDoutIndexType() const noexcept { return dout_.getStorageView().getIndexType(); }
    FTrainIndexType getLseIndexType() const noexcept { return lse_.getStorageView().getIndexType(); }
    FTrainIndexType getAttnSinkIndexType() const noexcept { return attn_sink_.getStorageView().getIndexType(); }
    FTrainIndexType getTopkIndicesIndexType() const noexcept { return topk_indices_.getStorageView().getIndexType(); }
    FTrainIndexType getTopkLengthIndexType() const noexcept { return topk_length_.getStorageView().getIndexType(); }
    FTrainIndexType getDqIndexType() const noexcept { return dq_.getStorageView().getIndexType(); }
    FTrainIndexType getDkvIndexType() const noexcept { return dkv_.getStorageView().getIndexType(); }
    FTrainIndexType getDsinkIndexType() const noexcept { return dsink_.getStorageView().getIndexType(); }

    // getTopkLengthAddress may be null: the compact (length-free) mode.
    void* getQAddress() const noexcept { return q_.getStorageView().getMemory(); }
    void* getKvAddress() const noexcept { return kv_.getStorageView().getMemory(); }
    void* getOutAddress() const noexcept { return out_.getStorageView().getMemory(); }
    void* getDoutAddress() const noexcept { return dout_.getStorageView().getMemory(); }
    void* getLseAddress() const noexcept { return lse_.getStorageView().getMemory(); }
    void* getAttnSinkAddress() const noexcept { return attn_sink_.getStorageView().getMemory(); }
    void* getTopkIndicesAddress() const noexcept { return topk_indices_.getStorageView().getMemory(); }
    void* getTopkLengthAddress() const noexcept { return topk_length_.getStorageView().getMemory(); }
    void* getDqAddress() const noexcept { return dq_.getStorageView().getMemory(); }
    void* getDkvAddress() const noexcept { return dkv_.getStorageView().getMemory(); }
    void* getDsinkAddress() const noexcept { return dsink_.getStorageView().getMemory(); }

    // Host scalar ports, following Gemm alpha/beta.
    FTrainNumericType getSoftmaxScaleNumericType() const noexcept {
        return softmax_scale_.getStorageView().getNumericType();
    }
    FTrainIndexType getSoftmaxScaleIndexType() const noexcept { return softmax_scale_.getStorageView().getIndexType(); }
    void* getSoftmaxScaleAddress() const noexcept { return softmax_scale_.getStorageView().getMemory(); }

    bool usesCachedBackward() const noexcept { return cached_backward_; }
    bool usesLowMemoryWorkspace() const noexcept { return low_memory_workspace_; }

    // Cache contract: encode every attribute and port's dtype, shape, layout,
    // presence and alignment used by isApplicable. Never encode raw addresses;
    // configure() binds fresh pointers after both cache hits and misses.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    static std::uint64_t dim(const Tensor& port, std::uint64_t axis) noexcept {
        return static_cast<std::uint64_t>(port.getStorageView().getDims()[axis]);
    }

    Tensor q_;
    Tensor kv_;
    Tensor out_;
    Tensor dout_;
    Tensor lse_;
    Tensor attn_sink_;
    Tensor topk_indices_;
    Tensor topk_length_;
    Tensor dq_;
    Tensor dkv_;
    Tensor dsink_;
    Tensor softmax_scale_;
    SparseAttnBwdAttributes attributes_;
    bool low_memory_workspace_ = false;
    bool cached_backward_      = true;
};

}  // namespace ftrain

#endif
