// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_FINDER_HPP_
#define FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sparse_indexer_predict/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct SparseIndexerPredictRegistrationOrderFinder final {
    static const char* getName() { return "SparseIndexerPredictRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseIndexerPredictProblem&, const Constraints&) {
        return {};
    }
};

}  // namespace ftrain

#endif
