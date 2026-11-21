// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_KDA_PROBLEM_HPP_
#define FTRAIN_FAMILY_KDA_PROBLEM_HPP_

#include <cstdint>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/operation/kda.hpp"
#include "flash_train/tensor.hpp"

namespace ftrain {

// The KDA problem dimensions: T packed tokens, H heads, K = key head dim, V = value head dim,
// N sequences (1 without cu_seqlens).
struct KdaShape {
    std::int64_t tokens;
    std::int64_t heads;
    std::int64_t key_dim;
    std::int64_t value_dim;
    std::int64_t seqs;
};

// Chunks of the packed layout: T / 64 for one sequence of T % 64 == 0 rows, else ceil(T / 64) + N.
inline std::int64_t kdaChunks(const KdaShape& s, bool has_cu) noexcept {
    return (!has_cu && s.tokens % 64 == 0) ? s.tokens / 64 : (s.tokens + 63) / 64 + (has_cu ? s.seqs : 1);
}

// Elements of the optional state port (ftrainKdaStateElements of flash_train.h).
inline std::int64_t kdaStateElements(const KdaShape& s, bool has_cu) noexcept {
    return s.heads * kdaChunks(s, has_cu) * 32768;
}

// Elements of the optional saved-h port (ftrainKdaSavedHElements of flash_train.h).
inline std::int64_t kdaSavedHElements(const KdaShape& s, bool has_cu) noexcept {
    return s.heads * kdaChunks(s, has_cu) * 16384;
}

// One KDA call's ports in packed THD layout, structurally validated by the constructors:
// every port is rank-consistent with
//   q, k [T, H, K]   v [T, H, V]   g [T, H, K]   beta [T, H]   a_log [H]   dt_bias [H, K] or [H*K]
//   cu_seqlens [N + 1]   q_rstd, k_rstd [T, H]   aqk, akk [T, H, 64]   o / d_o [T, H, V]
//   qn, kn [T, H, K] (optional, l2norm_qk only: the l2-normalized q / k the forward saves)
//   h [kdaSavedHElements(T, H, N, cu)] (optional, opaque saved recurrent state h)
//   state [kdaStateElements(T, H, N, cu)] (optional, opaque recurrent state; exclusive with h)
// and lives in device memory. Optional ports carry null memory when unused (their dims are
// then ignored): a_log/dt_bias (and their gradients) without gate_in_kernel, cu_seqlens for one
// sequence, q_rstd/k_rstd without l2norm_qk, qn/kn/h/state when not exchanged, and in the
// backward q/k when qn/kn are given. Value support (types, dims the kernels handle, device
// architecture) is the Primitives' isApplicable concern.
class KdaPortSet {
  public:
    const Tensor& get(std::size_t index) const noexcept { return tensors_[index]; }
    void* getAddress(std::size_t index) const noexcept { return tensors_[index].getStorageView().getMemory(); }
    FTrainNumericType getNumericType(std::size_t index) const noexcept {
        return tensors_[index].getStorageView().getNumericType();
    }
    FTrainIndexType getIndexType(std::size_t index) const noexcept {
        return tensors_[index].getStorageView().getIndexType();
    }
    bool hasMemory(std::size_t index) const noexcept { return getAddress(index) != nullptr; }
    std::size_t size() const noexcept { return tensors_.size(); }

  protected:
    std::vector<Tensor> tensors_;
};

class KdaForwardProblem final : public KdaPortSet {
  public:
    enum Port : std::size_t {
        kQ,
        kK,
        kV,
        kG,
        kBeta,
        kALog,
        kDtBias,
        kCuSeqlens,
        kO,
        kQRstd,
        kKRstd,
        kAqk,
        kAkk,
        kQn,
        kKn,
        kH,
        kState,
        kCount
    };

    // Throws Exception with FTRAIN_STATUS_INVALID_ARGUMENT on a structural violation.
    KdaForwardProblem(std::vector<Tensor> ports, KdaAttributes attributes);

    const KdaShape& getShape() const noexcept { return shape_; }
    const KdaAttributes& getAttributes() const noexcept { return attributes_; }

    // Every property that can change which Primitive applies (cache contract, see
    // GemmProblem::getProblemKey); the device is added by the engine.
    std::vector<std::uint64_t> getProblemKey() const;

  private:
    KdaShape shape_{};
    KdaAttributes attributes_;
};

class KdaBackwardProblem final : public KdaPortSet {
  public:
    enum Port : std::size_t {
        kDO,
        kQ,
        kK,
        kV,
        kG,
        kBeta,
        kALog,
        kDtBias,
        kCuSeqlens,
        kQRstd,
        kKRstd,
        kAqk,
        kAkk,
        kQn,
        kKn,
        kDq,
        kDk,
        kDv,
        kDg,
        kDBeta,
        kDALog,
        kDDtBias,
        kH,
        kState,
        kCount
    };

    KdaBackwardProblem(std::vector<Tensor> ports, KdaAttributes attributes);

    const KdaShape& getShape() const noexcept { return shape_; }
    const KdaAttributes& getAttributes() const noexcept { return attributes_; }

    std::vector<std::uint64_t> getProblemKey() const;

  private:
    KdaShape shape_{};
    KdaAttributes attributes_;
};

}  // namespace ftrain

#endif
