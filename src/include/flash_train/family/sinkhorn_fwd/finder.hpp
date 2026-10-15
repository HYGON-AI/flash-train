// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SINKHORN_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_SINKHORN_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sinkhorn_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct SinkhornFwdRegistrationOrderFinder final {
    static const char* getName() { return "SinkhornFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SinkhornFwdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
