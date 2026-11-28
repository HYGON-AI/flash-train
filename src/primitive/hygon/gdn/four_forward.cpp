// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

#include "flash_train/error.hpp"
#include "flash_train/primitive/hygon/arch.hpp"
#include "flash_train/primitive/hygon/gdn/four_forward.hpp"
#include "gfx936/common/flash_train_gdn_chunk_parallel.h"
#include "gfx936/common/flash_train_gdn_state_h.h"

namespace ftrain {
namespace {

std::size_t align256(std::size_t bytes) { return (bytes + 255) & ~std::size_t(255); }

struct WorkspaceLayout {
    std::size_t gc, w, u, vn, state, total;
};

WorkspaceLayout layout(int b, int s, int hv, int k, int v) {
    const std::size_t tokens = static_cast<std::size_t>(b) * s * hv;
    const std::size_t chunks = tokens / 64;
    WorkspaceLayout result{};
    std::size_t offset = 0;
    auto take = [&](std::size_t bytes) {
        const auto start = offset;
        offset += align256(bytes);
        return start;
    };
    result.gc = take(tokens * sizeof(float));
    result.w = take(tokens * k * sizeof(std::uint16_t));
    result.u = take(tokens * v * sizeof(std::uint16_t));
    result.vn = take(tokens * v * sizeof(std::uint16_t));
    result.state = take(chunks * k * v * sizeof(std::uint16_t));
    result.total = offset;
    return result;
}

}  // namespace

std::unique_ptr<PrimitiveBase> GdnFourForward::clone() const { return std::make_unique<GdnFourForward>(*this); }

Result GdnFourForward::isApplicable(const GdnForwardProblem& problem, const Constraints& constraints) const {
    if (!isGfx936(constraints)) return {FTRAIN_STATUS_UNSUPPORTED, "requires gfx936"};
    if (problem.hasH0() || problem.hasHt() || problem.getS() < 512 || problem.getS() % 64 != 0 ||
        (problem.getK() != 64 && problem.getK() != 128) ||
        (problem.getV() != 64 && problem.getV() != 128) ||
        problem.getB() > std::numeric_limits<int>::max() ||
        problem.getS() > std::numeric_limits<int>::max() ||
        problem.getH() > std::numeric_limits<int>::max() ||
        problem.getHV() > std::numeric_limits<int>::max() ||
        problem.getB() * problem.getHV() >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max()) /
                ((problem.getS() / 64) * (problem.getV() / 16))) {
        return {FTRAIN_STATUS_UNSUPPORTED, "four-kernel shape or state is unsupported"};
    }
    const auto valid = [](FTrainNumericType type, bool continuous, void* address,
                          std::uint32_t alignment, FTrainNumericType expected) {
        return type == expected && continuous && address != nullptr && alignment >= 16;
    };
#define CHECK(Name, Type) valid(problem.get##Name##NumericType(), problem.is##Name##Continuous(), \
                                problem.get##Name##Address(), problem.get##Name##Alignment(), Type)
    if (!CHECK(Q, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(K, FTRAIN_NUMERIC_TYPE_BF16) ||
        !CHECK(V, FTRAIN_NUMERIC_TYPE_BF16) || !CHECK(G, FTRAIN_NUMERIC_TYPE_FP32) ||
        !CHECK(Beta, FTRAIN_NUMERIC_TYPE_FP32) || !CHECK(Out, FTRAIN_NUMERIC_TYPE_BF16)) {
        return {FTRAIN_STATUS_UNSUPPORTED, "four-kernel dtype, layout or address is unsupported"};
    }
#undef CHECK
    return {};
}

void GdnFourForward::configure(const GdnForwardProblem& problem) {
    q_ = static_cast<const std::uint16_t*>(problem.getQAddress());
    k_ = static_cast<const std::uint16_t*>(problem.getKAddress());
    v_ = static_cast<const std::uint16_t*>(problem.getVAddress());
    g_ = static_cast<const float*>(problem.getGAddress());
    beta_ = static_cast<const float*>(problem.getBetaAddress());
    out_ = static_cast<std::uint16_t*>(problem.getOutAddress());
    b_ = static_cast<int>(problem.getB());
    s_ = static_cast<int>(problem.getS());
    h_ = static_cast<int>(problem.getH());
    hv_ = static_cast<int>(problem.getHV());
    kdim_ = static_cast<int>(problem.getK());
    vdim_ = static_cast<int>(problem.getV());
    scale_ = problem.getScale();
    workspace_bytes_ = layout(b_, s_, hv_, kdim_, vdim_).total;
}

void GdnFourForward::executeImpl(const Resources& resources) {
    if (reinterpret_cast<std::uintptr_t>(resources.getWorkspace()) % 16 != 0) {
        throw Exception(FTRAIN_STATUS_INVALID_ARGUMENT, "GDN workspace must be 16-byte aligned");
    }
    auto* base = static_cast<char*>(resources.getWorkspace());
    const auto l = layout(b_, s_, hv_, kdim_, vdim_);
    auto* gc = reinterpret_cast<float*>(base + l.gc);
    auto* w = reinterpret_cast<std::uint16_t*>(base + l.w);
    auto* u = reinterpret_cast<std::uint16_t*>(base + l.u);
    auto* vn = reinterpret_cast<std::uint16_t*>(base + l.vn);
    auto* state = reinterpret_cast<std::uint16_t*>(base + l.state);
    const auto stream = resources.getStream();
    const FlashTrainGDNIntraArgs intra{k_, v_, g_, beta_, w, u, gc, b_, s_, h_, hv_, kdim_, vdim_, nullptr};
    int status = flash_train_gdn_intra_bf16(&intra, stream);
    if (status == 0) {
        const FlashTrainGDNStateHArgs state_args{k_, w, u, gc, nullptr, state, vn, nullptr,
                                                  b_, s_, h_, hv_, kdim_, vdim_, 0, 1};
        status = flash_train_gdn_state_h_bf16(&state_args, stream);
    }
    if (status == 0) {
        const FlashTrainGDNChunkOArgs output_args{q_, k_, vn, state, gc, out_,
                                                   b_, s_, h_, hv_, kdim_, vdim_, scale_};
        status = flash_train_gdn_chunk_o_bf16(&output_args, stream);
    }
    if (status != 0) throw Exception(FTRAIN_STATUS_INTERNAL_ERROR, "GDN four-kernel launch failed (status %d)", status);
}

}  // namespace ftrain
