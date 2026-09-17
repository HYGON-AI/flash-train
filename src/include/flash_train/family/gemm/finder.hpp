// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#ifndef FTRAIN_FAMILY_GEMM_FINDER_HPP_
#define FTRAIN_FAMILY_GEMM_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/constraints.hpp"
#include "flash_train/family/gemm/problem.hpp"

namespace ftrain {

// The Gemm family's placeholder selection policy: it expresses no
// preference -- findCandidates returns no names, and the engine's
// registration-order walk supplies every applicable record. A real Finder
// policy provides the same static functions and returns the candidate
// names ranked best-first; OpsEngine consults the policies in pack order
// and skips names matching no registered record.
struct RegistrationOrderFinder final {
    static const char* getName() { return "RegistrationOrder"; }

    static std::vector<std::string> findCandidates(const GemmProblem&, const Constraints&) { return {}; }
};

}  // namespace ftrain

#endif
