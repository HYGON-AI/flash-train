// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/arch.hpp"
#include "flash_train/primitive/hygon/gdn2/chunk_backward_gfx938.hpp"
#include "gfx938/common/flash_train_gdn2_bwd.h"

namespace ftrain {

std::unique_ptr<PrimitiveBase> Gdn2ChunkBackwardGfx938::clone() const {
    return std::make_unique<Gdn2ChunkBackwardGfx938>(*this);
}

Result Gdn2ChunkBackwardGfx938::isApplicable(const Gdn2BackwardProblem& problem,
                                             const Constraints& constraints) const {
    if (!isGfx938(constraints)) return {FTRAIN_STATUS_UNSUPPORTED, "requires gfx938"};
    if (problem.getB() > std::numeric_limits<int>::max() ||
        problem.getS() > std::numeric_limits<int>::max() ||
        problem.getH() > std::numeric_limits<int>::max() ||
        problem.getHV() > std::numeric_limits<int>::max() ||
        problem.getK() > std::numeric_limits<int>::max() ||
        problem.getV() > std::numeric_limits<int>::max()) {
        return {FTRAIN_STATUS_UNSUPPORTED, "chunk backward shape is unsupported"};
    }
    // The archive's own backward gate (S >= 1, K, V <= 128, HV % H == 0;
    // other shapes padded internally) is the single source of truth and
    // keeps B, H, HV positive, which the product guard below divides by.
    if (flash_train_gdn2_chunk_backward_shape_supported(
            static_cast<int>(problem.getS()), static_cast<int>(problem.getH()),
            static_cast<int>(problem.getHV()), static_cast<int>(problem.getK()),
            static_cast<int>(problem.getV())) == 0) {
        return {FTRAIN_STATUS_UNSUPPORTED, "chunk backward shape is unsupported"};
    }
    if (problem.getB() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / problem.getHV()) {
        return {FTRAIN_STATUS_UNSUPPORTED, "chunk backward shape is unsupported"};
    }
    const auto valid = [](FTrainNumericType type, bool continuous, void* address,
                          std::uint32_t alignment, FTrainNumericType expected) {
        return type == expected && continuous && address != nullptr && alignment >= 16;
    };
#define CHECK(Name, Type) valid(problem.get##Name##NumericType(), problem.is##Name##Continuous(), \
                                problem.get##Name##Address(), problem.get##Name##Alignment(), Type)
    if (!CHECK(Q, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(K, FTRAIN_NUMERIC_TYPE_BF16) ||
        !CHECK(V, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(G, FTRAIN_NUMERIC_TYPE_FP32) ||
        !CHECK(BGate, FTRAIN_NUMERIC_TYPE_FP32) || !CHECK(W, FTRAIN_NUMERIC_TYPE_FP32) ||
        !CHECK(Do, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(DQ, FTRAIN_NUMERIC_TYPE_BF16) ||
        !CHECK(DK, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(DV, FTRAIN_NUMERIC_TYPE_BF16) ||
        !CHECK(DG, FTRAIN_NUMERIC_TYPE_FP32) || !CHECK(DBGate, FTRAIN_NUMERIC_TYPE_BF16) ||
        !CHECK(DW, FTRAIN_NUMERIC_TYPE_BF16)) {
        return {FTRAIN_STATUS_UNSUPPORTED, "chunk backward dtype, layout or address is unsupported"};
    }
#undef CHECK
    return {};
}

void Gdn2ChunkBackwardGfx938::configure(const Gdn2BackwardProblem& problem) {
    q_ = static_cast<const std::uint16_t*>(problem.getQAddress());
    k_ = static_cast<const std::uint16_t*>(problem.getKAddress());
    v_ = static_cast<const std::uint16_t*>(problem.getVAddress());
    g_ = static_cast<const float*>(problem.getGAddress());
    b_gate_ = static_cast<const float*>(problem.getBGateAddress());
    w_ = static_cast<const float*>(problem.getWAddress());
    do_ = static_cast<const std::uint16_t*>(problem.getDoAddress());
    dq_ = static_cast<std::uint16_t*>(problem.getDQAddress());
    dk_ = static_cast<std::uint16_t*>(problem.getDKAddress());
    dv_ = static_cast<std::uint16_t*>(problem.getDVAddress());
    dg_ = static_cast<float*>(problem.getDGAddress());
    db_ = static_cast<std::uint16_t*>(problem.getDBGateAddress());
    dw_ = static_cast<std::uint16_t*>(problem.getDWAddress());
    b_ = static_cast<int>(problem.getB());
    s_ = static_cast<int>(problem.getS());
    h_ = static_cast<int>(problem.getH());
    hv_ = static_cast<int>(problem.getHV());
    kdim_ = static_cast<int>(problem.getK());
    vdim_ = static_cast<int>(problem.getV());
    scale_ = problem.getScale();
    workspace_bytes_ = flash_train_gdn2_chunk_backward_workspace_bytes(b_, s_, h_, hv_, kdim_, vdim_);
}

void Gdn2ChunkBackwardGfx938::executeImpl(const Resources& resources) {
    if (reinterpret_cast<std::uintptr_t>(resources.getWorkspace()) % 16 != 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN2 backward workspace must be 16-byte aligned");
    }
    const int status = flash_train_gdn2_chunk_backward_bf16(
        q_, k_, v_, g_, b_gate_, w_, do_, nullptr, dq_, dk_, dv_, dg_, db_, dw_, nullptr,
        b_, s_, h_, hv_, kdim_, vdim_, scale_, resources.getWorkspace(),
        resources.getWorkspaceBytes(), resources.getStream());
    if (status != 0) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN2 chunk backward launch failed (status %d)", status);
    }
}

}  // namespace ftrain
