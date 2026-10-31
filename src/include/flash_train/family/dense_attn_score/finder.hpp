// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_ATTN_SCORE_FINDER_HPP_
#define FTRAIN_FAMILY_DENSE_ATTN_SCORE_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/dense_attn_score/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct DenseAttnScoreRegistrationOrderFinder final {
    static const char* getName() { return "DenseAttnScoreRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const DenseAttnScoreProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
