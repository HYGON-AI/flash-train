// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_INDEXER_SCORE_THD_FINDER_HPP_
#define FTRAIN_FAMILY_DENSE_INDEXER_SCORE_THD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/dense_indexer_score_thd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct DenseIndexerScoreThdRegistrationOrderFinder final {
    static const char* getName() { return "DenseIndexerScoreThdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const DenseIndexerScoreThdProblem&, const Constraints&) {
        return {};
    }
};

}  // namespace ftrain

#endif
