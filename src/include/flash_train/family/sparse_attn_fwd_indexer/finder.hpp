// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_FWD_INDEXER_FINDER_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_FWD_INDEXER_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sparse_attn_fwd_indexer/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct SparseAttnFwdIndexerRegistrationOrderFinder final {
    static const char* getName() { return "SparseAttnFwdIndexerRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseAttnFwdIndexerProblem&, const Constraints&) {
        return {};
    }
};

}  // namespace ftrain

#endif
