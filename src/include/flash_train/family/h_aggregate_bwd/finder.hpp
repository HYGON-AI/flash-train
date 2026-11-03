// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_AGGREGATE_BWD_FINDER_HPP_
#define FTRAIN_FAMILY_H_AGGREGATE_BWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/h_aggregate_bwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct HAggregateBwdRegistrationOrderFinder final {
    static const char* getName() { return "HAggregateBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const HAggregateBwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getXNumericType() != FTRAIN_NUMERIC_TYPE_FP32) ? "MixedFp32HAggregateBwd"
                                                                               : "Fp32HAggregateBwd";
                break;
            default: return {};
        }
        const auto preferred = (problem.getChannels() % 8 == 0 &&
                                (256 * problem.getStreams() + problem.getStreams()) * sizeof(float) <= 65536)
                                   ? std::string("Vectorized")
                                   : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "Vectorized"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
