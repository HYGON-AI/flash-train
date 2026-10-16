// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_TOPK_THD_FINDER_HPP_
#define FTRAIN_FAMILY_INDEXER_TOPK_THD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/indexer_topk_thd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registration-order selection: this family has a single implementation.
struct IndexerTopKThdRegistrationOrderFinder final {
    static const char* getName() { return "IndexerTopKThdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const IndexerTopKThdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
