// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_COMPACTIFY_FINDER_HPP_
#define FTRAIN_FAMILY_COMPACTIFY_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/compactify/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct CompactifyRegistrationOrderFinder final {
    static const char* getName() { return "CompactifyRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const CompactifyProblem& problem, const Constraints&) {
        const char* base = "Compactify";
        const auto preferred =
            (problem.getSelectionWidth() > 256 || (problem.getSelectionWidth() == 256 && problem.getNumRows() >= 4096))
                ? std::string("Large")
                : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "Large"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
