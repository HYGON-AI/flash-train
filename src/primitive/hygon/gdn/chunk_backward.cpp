// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/arch.hpp"
#include "flash_train/primitive/hygon/gdn/chunk_backward.hpp"
#include "gfx936/common/flash_train_gdn_bwd_entry.h"

namespace ftrain {

int transposeGdnState(const float* source, float* target, int b, int hv, int k, int v,
                      bool to_v_first, FTrainStream stream);

std::unique_ptr<PrimitiveBase> GdnChunkBackward::clone() const {
    return std::make_unique<GdnChunkBackward>(*this);
}

Result GdnChunkBackward::isApplicable(const GdnBackwardProblem& problem, const Constraints& constraints) const {
    if (!isGfx936(constraints)) return {FTRAIN_STATUS_UNSUPPORTED, "requires gfx936"};
    if (problem.getS() < 512 || problem.getS() % 64 != 0 ||
        (problem.getK() != 64 && problem.getK() != 128) ||
        (problem.getV() != 64 && problem.getV() != 128) ||
        problem.getB() > std::numeric_limits<int>::max() ||
        problem.getS() > std::numeric_limits<int>::max() ||
        problem.getH() > std::numeric_limits<int>::max() ||
        problem.getHV() > std::numeric_limits<int>::max() ||
        problem.getB() * problem.getHV() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) /
                ((problem.getS() / 64) * (problem.getV() / 16))) {
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
        !CHECK(Beta, FTRAIN_NUMERIC_TYPE_FP32) || !CHECK(Do, FTRAIN_NUMERIC_TYPE_BF16) ||
        !CHECK(DQ, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(DK, FTRAIN_NUMERIC_TYPE_BF16) ||
        !CHECK(DV, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(DG, FTRAIN_NUMERIC_TYPE_FP32) ||
        !CHECK(DBeta, FTRAIN_NUMERIC_TYPE_FP32) ||
        (problem.hasH0() && !CHECK(H0, FTRAIN_NUMERIC_TYPE_FP32)) ||
        (problem.hasDht() && !CHECK(Dht, FTRAIN_NUMERIC_TYPE_FP32)) ||
        (problem.hasDh0() && !CHECK(Dh0, FTRAIN_NUMERIC_TYPE_FP32))) {
        return {FTRAIN_STATUS_UNSUPPORTED, "chunk backward dtype, layout or address is unsupported"};
    }
#undef CHECK
    return {};
}

void GdnChunkBackward::configure(const GdnBackwardProblem& problem) {
    q_ = static_cast<const std::uint16_t*>(problem.getQAddress());
    k_ = static_cast<const std::uint16_t*>(problem.getKAddress());
    v_ = static_cast<const std::uint16_t*>(problem.getVAddress());
    g_ = static_cast<const float*>(problem.getGAddress());
    beta_ = static_cast<const float*>(problem.getBetaAddress());
    do_ = static_cast<const std::uint16_t*>(problem.getDoAddress());
    h0_ = problem.hasH0() ? static_cast<const float*>(problem.getH0Address()) : nullptr;
    dht_ = problem.hasDht() ? static_cast<const float*>(problem.getDhtAddress()) : nullptr;
    dq_ = static_cast<std::uint16_t*>(problem.getDQAddress());
    dk_ = static_cast<std::uint16_t*>(problem.getDKAddress());
    dv_ = static_cast<std::uint16_t*>(problem.getDVAddress());
    dg_ = static_cast<float*>(problem.getDGAddress());
    dbeta_ = static_cast<float*>(problem.getDBetaAddress());
    dh0_ = problem.hasDh0() ? static_cast<float*>(problem.getDh0Address()) : nullptr;
    b_ = static_cast<int>(problem.getB());
    s_ = static_cast<int>(problem.getS());
    h_ = static_cast<int>(problem.getH());
    hv_ = static_cast<int>(problem.getHV());
    kdim_ = static_cast<int>(problem.getK());
    vdim_ = static_cast<int>(problem.getV());
    scale_ = problem.getScale();
    const FlashTrainGDNChunkShape shape{b_, s_, h_, hv_, kdim_, vdim_, 0};
    core_bytes_ = flash_train_gdn_chunk_backward_workspace_bytes(shape);
    const auto state_bytes = static_cast<std::uint64_t>(b_) * hv_ * kdim_ * vdim_ * sizeof(float);
    const auto aligned_state = (state_bytes + 255) & ~std::uint64_t(255);
    workspace_bytes_ = core_bytes_;
    if (dht_) {
        dht_offset_ = workspace_bytes_;
        workspace_bytes_ += aligned_state;
    }
    if (dh0_) {
        dh0_offset_ = workspace_bytes_;
        workspace_bytes_ += aligned_state;
    }
}

void GdnChunkBackward::executeImpl(const Resources& resources) {
    if (reinterpret_cast<std::uintptr_t>(resources.getWorkspace()) % 16 != 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN backward workspace must be 16-byte aligned");
    }
    auto* base = static_cast<char*>(resources.getWorkspace());
    auto* dht_v_first = dht_ ? reinterpret_cast<float*>(base + dht_offset_) : nullptr;
    auto* dh0_v_first = dh0_ ? reinterpret_cast<float*>(base + dh0_offset_) : nullptr;
    const auto stream = resources.getStream();
    if (dht_ && transposeGdnState(dht_, dht_v_first, b_, hv_, kdim_, vdim_, true, stream) != 0) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN dht transpose launch failed");
    }
    const FlashTrainGDNChunkBackwardArgs args{
        q_, k_, v_, g_, beta_, do_, h0_, dht_v_first,
        dq_, dk_, dv_, dg_, dbeta_, dh0_v_first,
        {b_, s_, h_, hv_, kdim_, vdim_, 0}, scale_,
        resources.getWorkspace(), static_cast<std::size_t>(core_bytes_)};
    const int status = flash_train_gdn_chunk_bf16_backward_launch(&args, stream);
    if (status != FLASH_TRAIN_GDN_CHUNK_SUCCESS) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN chunk backward launch failed (status %d)", status);
    }
    if (dh0_ && transposeGdnState(dh0_v_first, dh0_, b_, hv_, kdim_, vdim_, false, stream) != 0) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN dh0 transpose launch failed");
    }
}

}  // namespace ftrain
