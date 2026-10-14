// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_SESSION_DISTRIBUTED_HPP_
#define FTRAIN_SESSION_DISTRIBUTED_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "flash_train/common.h"
#include "flash_train/tensor.hpp"

namespace ftrain {

class AGGroupGemmProblem;
class MoEReduceRSProblem;

// Plain-data metadata shared with Primitive applicability checks.  A concrete
// distributed backend owns communicators and device resources; this core
// session interface exposes only values and execution.
struct AGGroupGemmSessionInfo final {
    std::uint64_t session_id{};
    FTrainAGRoute route{FTRAIN_AG_ROUTE_INVALID};
    std::int32_t world_size{};
    std::int32_t rank{};
    std::int32_t nnodes{1};
    std::uint64_t full_m{};
    std::uint64_t n_dim{};
    std::uint64_t k_dim{};
    std::uint64_t num_experts{};
    std::uint64_t topk{};
    std::vector<std::int64_t> rank_m_sizes;
    FTrainNumericType input_type{FTRAIN_NUMERIC_TYPE_INVALID};
    FTrainNumericType output_type{FTRAIN_NUMERIC_TYPE_INVALID};
};

struct MoEReduceRSSessionInfo final {
    std::uint64_t session_id{};
    FTrainMoERSRoute route{FTRAIN_MOERS_ROUTE_INVALID};
    FTrainMoERSGemmVariant gemm_variant{FTRAIN_MOERS_GEMM_INVALID};
    std::int32_t world_size{};
    std::int32_t rank{};
    std::int32_t nnodes{1};
    std::uint64_t max_ntokens{};
    std::uint64_t n_dim{};
    std::uint64_t k_dim{};
    std::uint64_t num_experts{};
    std::uint64_t topk{};
    std::vector<std::int64_t> rank_m_sizes;
    FTrainNumericType input_type{FTRAIN_NUMERIC_TYPE_INVALID};
    std::int32_t n_chunks{};
};

class AGGroupGemmSession {
  public:
    virtual ~AGGroupGemmSession() = default;

    virtual const AGGroupGemmSessionInfo& info() const noexcept = 0;
    virtual void execute(const AGGroupGemmProblem& problem, FTrainStream stream) = 0;
};

class MoEReduceRSSession {
  public:
    virtual ~MoEReduceRSSession() = default;

    virtual const MoEReduceRSSessionInfo& info() const noexcept = 0;
    virtual void execute(const MoEReduceRSProblem& problem, FTrainStream stream) = 0;
};

// Registry is filled by the optional Torch backend; core Primitives consume it.
class DistributedSessionRegistry final {
  public:
    static DistributedSessionRegistry& get();

    void registerAGGroupGemm(std::uint64_t session_id, std::shared_ptr<AGGroupGemmSession> session);
    void registerMoEReduceRS(std::uint64_t session_id, std::shared_ptr<MoEReduceRSSession> session);

    std::shared_ptr<AGGroupGemmSession> getAGGroupGemm(std::uint64_t session_id) const;
    std::shared_ptr<MoEReduceRSSession> getMoEReduceRS(std::uint64_t session_id) const;

    void eraseAGGroupGemm(std::uint64_t session_id);
    void eraseMoEReduceRS(std::uint64_t session_id);

  private:
    DistributedSessionRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::uint64_t, std::shared_ptr<AGGroupGemmSession>> ag_sessions_;
    std::unordered_map<std::uint64_t, std::shared_ptr<MoEReduceRSSession>> moe_sessions_;
};

}  // namespace ftrain

#endif
