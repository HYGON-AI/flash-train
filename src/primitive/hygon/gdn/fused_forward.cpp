// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/arch.hpp"
#include "flash_train/primitive/hygon/gdn/fused_forward.hpp"
#include "gfx936/common/flash_train_gdn_chunk.h"

namespace ftrain {

const char* GdnFusedForward::getName() const noexcept {
    return type_ == FTRAIN_NUMERIC_TYPE_BF16 ? "GdnFusedForwardBf16" : "GdnFusedForwardFp32";
}

std::unique_ptr<PrimitiveBase> GdnFusedForward::clone() const { return std::make_unique<GdnFusedForward>(*this); }

Result GdnFusedForward::isApplicable(const GdnForwardProblem& problem, const Constraints& constraints) const {
    if (!isGfx936(constraints)) return {FTRAIN_STATUS_UNSUPPORTED, "requires gfx936"};
    if (problem.getS() % 32 != 0 || problem.getK() > 128 || problem.getV() % 16 != 0 ||
        problem.getB() > std::numeric_limits<int>::max() ||
        problem.getS() > std::numeric_limits<int>::max() ||
        problem.getH() > std::numeric_limits<int>::max() ||
        problem.getHV() > std::numeric_limits<int>::max() ||
        problem.getK() > std::numeric_limits<int>::max() ||
        problem.getV() > std::numeric_limits<int>::max() ||
        problem.getB() > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / problem.getHV() ||
        problem.getB() * problem.getHV() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / (problem.getV() / 16)) {
        return {FTRAIN_STATUS_UNSUPPORTED, "fused chunk shape is unsupported"};
    }
    const auto valid = [](FTrainNumericType type, bool continuous, void* address,
                          std::uint32_t alignment, FTrainNumericType expected) {
        return type == expected && continuous && address != nullptr &&
               alignment >= (expected == FTRAIN_NUMERIC_TYPE_FP32 ? 4U : 2U);
    };
#define CHECK(Name, Type) valid(problem.get##Name##NumericType(), problem.is##Name##Continuous(), \
                                problem.get##Name##Address(), problem.get##Name##Alignment(), Type)
    if (!CHECK(Q, type_) || !CHECK(K, type_) || !CHECK(V, type_) ||
        !CHECK(G, FTRAIN_NUMERIC_TYPE_FP32) || !CHECK(Beta, FTRAIN_NUMERIC_TYPE_FP32) ||
        !CHECK(Out, type_) ||
        (problem.hasH0() && !CHECK(H0, FTRAIN_NUMERIC_TYPE_FP32)) ||
        (problem.hasHt() && !CHECK(Ht, FTRAIN_NUMERIC_TYPE_FP32))) {
        return {FTRAIN_STATUS_UNSUPPORTED, "fused chunk dtype, layout or address is unsupported"};
    }
#undef CHECK
    return {};
}

void GdnFusedForward::configure(const GdnForwardProblem& problem) {
    q_ = problem.getQAddress();
    k_ = problem.getKAddress();
    v_ = problem.getVAddress();
    g_ = static_cast<const float*>(problem.getGAddress());
    beta_ = static_cast<const float*>(problem.getBetaAddress());
    h0_ = problem.hasH0() ? static_cast<const float*>(problem.getH0Address()) : nullptr;
    out_ = problem.getOutAddress();
    ht_ = problem.hasHt() ? static_cast<float*>(problem.getHtAddress()) : nullptr;
    b_ = static_cast<int>(problem.getB());
    s_ = static_cast<int>(problem.getS());
    h_ = static_cast<int>(problem.getH());
    hv_ = static_cast<int>(problem.getHV());
    kdim_ = static_cast<int>(problem.getK());
    vdim_ = static_cast<int>(problem.getV());
    scale_ = problem.getScale();
}

void GdnFusedForward::executeImpl(const Resources& resources) {
    const FlashTrainGDNChunkShape shape{b_, s_, h_, hv_, kdim_, vdim_, 32};
    int status;
    if (type_ == FTRAIN_NUMERIC_TYPE_BF16) {
        status = flash_train_gdn_chunk_bf16_forward_launch(
            static_cast<const std::uint16_t*>(q_), static_cast<const std::uint16_t*>(k_),
            static_cast<const std::uint16_t*>(v_), g_, beta_, h0_, static_cast<std::uint16_t*>(out_), ht_,
            nullptr, shape, scale_, resources.getStream());
    } else {
        status = flash_train_gdn_chunk_fp32_forward_launch(
            static_cast<const float*>(q_), static_cast<const float*>(k_), static_cast<const float*>(v_),
            g_, beta_, h0_, static_cast<float*>(out_), ht_, nullptr, shape, scale_, resources.getStream());
    }
    if (status != FLASH_TRAIN_GDN_CHUNK_SUCCESS) {
        throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN fused forward launch failed (status %d)", status);
    }
}

}  // namespace ftrain
