// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_KDA_FINDER_HPP_
#define FTRAIN_FAMILY_KDA_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/constraints.hpp"
#include "flash_train/family/kda/problem.hpp"

namespace ftrain {

// The KDA families' selection policy: no preference beyond the registration order (the
// recommended implementation is registered first).
struct KdaRegistrationOrderFinder final {
    static const char* getName() { return "KdaRegistrationOrder"; }

    static std::vector<std::string> findCandidates(const KdaForwardProblem&, const Constraints&) { return {}; }
    static std::vector<std::string> findCandidates(const KdaBackwardProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
