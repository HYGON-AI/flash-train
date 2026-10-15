// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_INDEXER_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/indexer_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct IndexerFwdRegistrationOrderFinder final {
    static const char* getName() { return "IndexerFwdRegistrationOrder"; }

    static std::vector<std::string> findCandidates(const IndexerFwdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
