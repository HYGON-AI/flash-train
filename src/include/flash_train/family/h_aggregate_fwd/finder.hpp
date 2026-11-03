// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_AGGREGATE_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_H_AGGREGATE_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/h_aggregate_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct HAggregateFwdRegistrationOrderFinder final {
    static const char* getName() { return "HAggregateFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const HAggregateFwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP32) ? "MixedFp32HAggregateFwd"
                                                                               : "Fp32HAggregateFwd";
                break;
            default: return {};
        }
        const auto preferred = (problem.getChannels() % 8 == 0) ? std::string("Vectorized") : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "Vectorized"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
