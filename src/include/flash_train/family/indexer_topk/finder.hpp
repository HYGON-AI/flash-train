// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_TOPK_FINDER_HPP_
#define FTRAIN_FAMILY_INDEXER_TOPK_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/indexer_topk/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registration-order selection: this family has a single implementation, so
// the policy only exists so a host-specific implementation can be ranked
// later without touching the engine.
struct IndexerTopKRegistrationOrderFinder final {
    static const char* getName() { return "IndexerTopKRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const IndexerTopKProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
