// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_MOE_REDUCE_RS_FINDER_HPP_
#define FTRAIN_FAMILY_MOE_REDUCE_RS_FINDER_HPP_

#include <flash_train/constraints.hpp>
#include <flash_train/family/moe_reduce_rs/problem.hpp>

#include <string>
#include <vector>

namespace ftrain {

struct MoEReduceRSRouteFinder final {
    static const char* getName() { return "HygonMoEReduceRSRoute"; }

    static std::vector<std::string> findCandidates(const MoEReduceRSProblem& problem, const Constraints&);
};

}  // namespace ftrain

#endif
