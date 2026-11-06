// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#include "distributed_session.hpp"

#include <flash_train/family/ag_group_gemm/problem.hpp>
#include <flash_train/family/moe_reduce_rs/problem.hpp>
#include <flash_train/session/distributed.hpp>

#include <interface/ag_group_gemm_interface.h>
#include <interface/moe_reduce_rs_interface.h>

#include <ATen/ATen.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <c10/hip/HIPFunctions.h>

#include <atomic>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftrain::compat {
namespace {

c10::ScalarType supportedDtype(c10::ScalarType dtype) {
    if (dtype != c10::kHalf && dtype != c10::kBFloat16) {
        throw std::invalid_argument("Hygon distributed operators support FP16 and BF16 only");
    }
    return dtype;
}

FTrainNumericType numericType(c10::ScalarType dtype) {
    return dtype == c10::kHalf ? FTRAIN_NUMERIC_TYPE_FP16 : FTRAIN_NUMERIC_TYPE_BF16;
}

const char *agRouteName(FTrainAGRoute route) {
    switch (route) {
        case FTRAIN_AG_ROUTE_NATIVE_NK_A2A: return "native_nk_a2a";
        case FTRAIN_AG_ROUTE_NATIVE_KN_A2A: return "native_kn_a2a";
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED: return "native_kn_ring_fixed";
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING: return "native_kn_ring";
        case FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK: return "native_kn_ring_readymask";
        default: throw std::invalid_argument("invalid AGGroupGemm route");
    }
}

const char *moeRouteName(FTrainMoERSRoute route) {
    switch (route) {
        case FTRAIN_MOERS_ROUTE_A2A: return "a2a";
        case FTRAIN_MOERS_ROUTE_A2A_RAGGED_SIGNAL: return "a2a_ragged_signal";
        default: throw std::invalid_argument("invalid MoEReduceRS route");
    }
}

const char *moeVariantName(FTrainMoERSGemmVariant variant) {
    switch (variant) {
        case FTRAIN_MOERS_GEMM_NK_DIRECT: return "nk_direct";
        case FTRAIN_MOERS_GEMM_KN_DIRECT: return "kn_direct";
        default: throw std::invalid_argument("invalid MoEReduceRS gemm variant");
    }
}

at::Tensor atTensor(const ftrain::Tensor& tensor) {
    const auto& view = tensor.getStorageView();
    c10::ScalarType dtype;
    switch (view.getNumericType()) {
        case FTRAIN_NUMERIC_TYPE_FP16: dtype = c10::kHalf; break;
        case FTRAIN_NUMERIC_TYPE_BF16: dtype = c10::kBFloat16; break;
        case FTRAIN_NUMERIC_TYPE_INT32: dtype = c10::kInt; break;
        case FTRAIN_NUMERIC_TYPE_INT64: dtype = c10::kLong; break;
        default: throw std::invalid_argument("unsupported Hygon distributed tensor dtype");
    }
    const auto options = at::TensorOptions()
                             .dtype(dtype)
                             .device(at::kCUDA)
                             .device_index(c10::hip::current_device());
    const std::vector<int64_t> strides(view.getStrides().begin(), view.getStrides().end());
    return at::from_blob(view.getMemory(), view.getDims(), strides, [](void*) {}, options);
}

class TorchAGGroupGemmSession final : public ftrain::AGGroupGemmSession {
  public:
    TorchAGGroupGemmSession(std::uint64_t session_id, FTrainAGRoute route,
                            c10::intrusive_ptr<c10d::ProcessGroup> group, int32_t nnodes, int32_t full_m,
                            int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk, c10::ScalarType dtype,
                            std::vector<int64_t> rank_m_sizes, int32_t capacity_full_m)
        : info_(makeInfo(session_id, route, group, nnodes, full_m, n_dim, k_dim, num_experts, topk, dtype,
                         rank_m_sizes)),
          op_(std::move(group), nnodes, full_m, static_cast<int32_t>(n_dim), static_cast<int32_t>(k_dim),
              static_cast<int32_t>(num_experts), static_cast<int32_t>(topk), dtype, dtype, false,
              rank_m_sizes, capacity_full_m, agRouteName(route)) {}

    const ftrain::AGGroupGemmSessionInfo& info() const noexcept override { return info_; }

    void execute(const ftrain::AGGroupGemmProblem& problem, FTrainStream stream) override {
        c10::hip::HIPStreamGuardMasqueradingAsCUDA guard(
            c10::hip::getStreamFromExternalMasqueradingAsCUDA(stream, c10::hip::current_device()).unwrap());
        static_cast<void>(op_.forward(atTensor(problem.getInput()), atTensor(problem.getWeight()),
                                      atTensor(problem.getTopkIds()), atTensor(problem.getOutput())));
    }

  private:
    static ftrain::AGGroupGemmSessionInfo makeInfo(
        std::uint64_t session_id, FTrainAGRoute route, const c10::intrusive_ptr<c10d::ProcessGroup>& group,
        int32_t nnodes, int32_t full_m, int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk,
        c10::ScalarType dtype, const std::vector<int64_t>& rank_m_sizes) {
        ftrain::AGGroupGemmSessionInfo info;
        info.session_id = session_id;
        info.route = route;
        info.world_size = static_cast<int32_t>(group->getSize());
        info.rank = static_cast<int32_t>(group->getRank());
        info.nnodes = nnodes;
        info.full_m = static_cast<std::uint64_t>(full_m);
        info.n_dim = static_cast<std::uint64_t>(n_dim);
        info.k_dim = static_cast<std::uint64_t>(k_dim);
        info.num_experts = static_cast<std::uint64_t>(num_experts);
        info.topk = static_cast<std::uint64_t>(topk);
        info.rank_m_sizes = rank_m_sizes;
        info.input_type = numericType(dtype);
        info.output_type = numericType(dtype);
        return info;
    }

    ftrain::AGGroupGemmSessionInfo info_;
    ::bytedance::hygon_vendor::AGGroupGemm op_;
};

class TorchMoEReduceRSSession final : public ftrain::MoEReduceRSSession {
  public:
    TorchMoEReduceRSSession(std::uint64_t session_id, FTrainMoERSRoute route, FTrainMoERSGemmVariant gemm_variant,
                            c10::intrusive_ptr<c10d::ProcessGroup> group, int32_t nnodes, int32_t max_ntokens,
                            int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk, c10::ScalarType dtype,
                            const std::vector<int64_t>& rank_m_sizes, int32_t n_chunks)
        : info_(makeInfo(session_id, route, gemm_variant, group, nnodes, max_ntokens, n_dim, k_dim, num_experts,
                         topk, dtype, rank_m_sizes, n_chunks)),
          op_(std::move(group), nnodes, max_ntokens, static_cast<int32_t>(n_dim), static_cast<int32_t>(k_dim),
              static_cast<int32_t>(num_experts), static_cast<int32_t>(topk), dtype, n_chunks, moeRouteName(route),
              moeVariantName(gemm_variant)) {
        if (!rank_m_sizes.empty()) {
            op_.set_rank_m_sizes(static_cast<int32_t>(info_.max_ntokens), info_.rank_m_sizes);
        }
    }

    const ftrain::MoEReduceRSSessionInfo& info() const noexcept override { return info_; }

    void execute(const ftrain::MoEReduceRSProblem& problem, FTrainStream stream) override {
        c10::hip::HIPStreamGuardMasqueradingAsCUDA guard(
            c10::hip::getStreamFromExternalMasqueradingAsCUDA(stream, c10::hip::current_device()).unwrap());
        static_cast<void>(op_.run_a2a(atTensor(problem.getInput()), atTensor(problem.getWeight()),
                                      atTensor(problem.getChosenExperts()), atTensor(problem.getExpertWeight()),
                                      atTensor(problem.getOutput())));
    }

  private:
    static ftrain::MoEReduceRSSessionInfo makeInfo(
        std::uint64_t session_id, FTrainMoERSRoute route, FTrainMoERSGemmVariant gemm_variant,
        const c10::intrusive_ptr<c10d::ProcessGroup>& group, int32_t nnodes, int32_t max_ntokens, int32_t n_dim,
        int32_t k_dim, int32_t num_experts, int32_t topk, c10::ScalarType dtype,
        const std::vector<int64_t>& rank_m_sizes, int32_t n_chunks) {
        ftrain::MoEReduceRSSessionInfo info;
        info.session_id = session_id;
        info.route = route;
        info.gemm_variant = gemm_variant;
        info.world_size = static_cast<int32_t>(group->getSize());
        info.rank = static_cast<int32_t>(group->getRank());
        info.nnodes = nnodes;
        info.max_ntokens = static_cast<std::uint64_t>(max_ntokens);
        info.n_dim = static_cast<std::uint64_t>(n_dim);
        info.k_dim = static_cast<std::uint64_t>(k_dim);
        info.num_experts = static_cast<std::uint64_t>(num_experts);
        info.topk = static_cast<std::uint64_t>(topk);
        info.rank_m_sizes = rank_m_sizes;
        info.input_type = numericType(dtype);
        info.n_chunks = n_chunks;
        return info;
    }

    ftrain::MoEReduceRSSessionInfo info_;
    ::bytedance::hygon_vendor::MoEReduceRS op_;
};

std::uint64_t nextSessionId() {
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

std::uint64_t createAGGroupGemmSession(
    const std::string& route, c10::intrusive_ptr<c10d::ProcessGroup> group, int32_t nnodes, int32_t full_m,
    int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk, c10::ScalarType dtype,
    const std::vector<int64_t>& rank_m_sizes, int32_t capacity_full_m) {
    if (group == nullptr) { throw std::invalid_argument("AGGroupGemm requires an ETP ProcessGroup"); }
    std::string normalized_route = route.empty() || route == "auto" ? "native_nk_a2a" : route;

    FTrainAGRoute route_id;
    if (normalized_route == "native_nk_a2a") { route_id = FTRAIN_AG_ROUTE_NATIVE_NK_A2A; }
    else if (normalized_route == "native_kn_a2a") { route_id = FTRAIN_AG_ROUTE_NATIVE_KN_A2A; }
    else if (normalized_route == "native_kn_ring_fixed") { route_id = FTRAIN_AG_ROUTE_NATIVE_KN_RING_FIXED; }
    else if (normalized_route == "native_kn_ring") { route_id = FTRAIN_AG_ROUTE_NATIVE_KN_RING; }
    else if (normalized_route == "native_kn_ring_readymask") { route_id = FTRAIN_AG_ROUTE_NATIVE_KN_RING_READY_MASK; }
    else { throw std::invalid_argument("unsupported AGGroupGemm route: " + normalized_route); }

    const auto session_id = nextSessionId();
    auto session = std::make_shared<TorchAGGroupGemmSession>(
        session_id, route_id, std::move(group), nnodes, full_m, n_dim, k_dim, num_experts, topk, supportedDtype(dtype),
        std::move(rank_m_sizes), capacity_full_m);
    ftrain::DistributedSessionRegistry::get().registerAGGroupGemm(session_id, std::move(session));
    return session_id;
}

std::uint64_t createMoEReduceRSSession(
    const std::string& route, const std::string& gemm_variant, c10::intrusive_ptr<c10d::ProcessGroup> group,
    int32_t nnodes, int32_t max_ntokens, int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk,
    c10::ScalarType dtype, const std::vector<int64_t>& rank_m_sizes, int32_t n_chunks) {
    if (group == nullptr) { throw std::invalid_argument("MoEReduceRS requires an ETP ProcessGroup"); }

    FTrainMoERSRoute route_id;
    if (route == "a2a") { route_id = FTRAIN_MOERS_ROUTE_A2A; }
    else if (route == "a2a_ragged_signal") { route_id = FTRAIN_MOERS_ROUTE_A2A_RAGGED_SIGNAL; }
    else { throw std::invalid_argument("unsupported MoEReduceRS route: " + route); }

    std::string normalized_route = route.empty() ? "a2a_ragged_signal" : route;
    std::string normalized_variant = gemm_variant.empty() || gemm_variant == "auto" ? "nk_direct" : gemm_variant;

    FTrainMoERSGemmVariant variant_id;
    if (normalized_variant == "nk_direct") { variant_id = FTRAIN_MOERS_GEMM_NK_DIRECT; }
    else if (normalized_variant == "kn_direct") { variant_id = FTRAIN_MOERS_GEMM_KN_DIRECT; }
    else { throw std::invalid_argument("unsupported MoEReduceRS gemm variant: " + gemm_variant); }

    const auto session_id = nextSessionId();
    auto session = std::make_shared<TorchMoEReduceRSSession>(
        session_id, route_id, variant_id, std::move(group), nnodes, max_ntokens, n_dim, k_dim, num_experts, topk,
        supportedDtype(dtype), std::move(rank_m_sizes), n_chunks);
    ftrain::DistributedSessionRegistry::get().registerMoEReduceRS(session_id, std::move(session));
    return session_id;
}

}  // namespace ftrain::compat
