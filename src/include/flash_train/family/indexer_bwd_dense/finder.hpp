// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_BWD_DENSE_FINDER_HPP_
#define FTRAIN_FAMILY_INDEXER_BWD_DENSE_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/indexer_bwd_dense/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct IndexerBwdDenseRegistrationOrderFinder final {
    static const char* getName() { return "IndexerBwdDenseRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const IndexerBwdDenseProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
