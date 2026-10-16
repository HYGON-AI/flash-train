// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_POST_BDA_BWD_FINDER_HPP_
#define FTRAIN_FAMILY_H_POST_BDA_BWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/h_post_bda_bwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
struct HPostBdaBwdRegistrationOrderFinder final {
    static const char* getName() { return "HPostBdaBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const HPostBdaBwdProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
