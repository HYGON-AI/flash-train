// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_DISTRIBUTED_SESSION_HPP_
#define FTRAIN_DISTRIBUTED_SESSION_HPP_

#include <flash_train/session/distributed.hpp>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ftrain::compat {

std::uint64_t createAGGroupGemmSession(
    const std::string& route, c10::intrusive_ptr<c10d::ProcessGroup> group, int32_t nnodes, int32_t full_m,
    int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk, c10::ScalarType dtype,
    const std::vector<int64_t>& rank_m_sizes, int32_t capacity_full_m);

std::uint64_t createMoEReduceRSSession(
    const std::string& route, const std::string& gemm_variant, c10::intrusive_ptr<c10d::ProcessGroup> group,
    int32_t nnodes, int32_t max_ntokens, int32_t n_dim, int32_t k_dim, int32_t num_experts, int32_t topk,
    c10::ScalarType dtype, const std::vector<int64_t>& rank_m_sizes, int32_t n_chunks);

}  // namespace ftrain::compat

#endif
