// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_AG_GROUP_GEMM_FINDER_HPP_
#define FTRAIN_FAMILY_AG_GROUP_GEMM_FINDER_HPP_

#include <flash_train/constraints.hpp>
#include <flash_train/family/ag_group_gemm/problem.hpp>

#include <string>
#include <vector>

namespace ftrain {

struct AGGroupGemmRouteFinder final {
    static const char* getName() { return "HygonAGGroupGemmRoute"; }

    static std::vector<std::string> findCandidates(const AGGroupGemmProblem& problem, const Constraints&);
};

}  // namespace ftrain

#endif
