// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SINKHORN_BWD_FINDER_HPP_
#define FTRAIN_FAMILY_SINKHORN_BWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sinkhorn_bwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct SinkhornBwdRegistrationOrderFinder final {
    static const char* getName() { return "SinkhornBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SinkhornBwdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
