// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_TARGET_FINDER_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_TARGET_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sparse_attn_target/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct SparseAttnTargetRegistrationOrderFinder final {
    static const char* getName() { return "SparseAttnTargetRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseAttnTargetProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
