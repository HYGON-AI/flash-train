// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_H_POST_BDA_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_H_POST_BDA_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/h_post_bda_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct HPostBdaFwdRegistrationOrderFinder final {
    static const char* getName() { return "HPostBdaFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const HPostBdaFwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16HPostBdaFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16HPostBdaFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32: base = "Fp32HPostBdaFwd"; break;
            default: return {};
        }
        const auto preferred =
            (problem.getChannels() % 8 == 0 && (problem.getStreams() == 2 || problem.getStreams() == 4 ||
                                                problem.getStreams() == 8 || problem.getStreams() == 16))
                ? std::string("Vectorized")
                : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "Vectorized"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        if (problem.getStreams() != 2 && problem.getStreams() != 4 && problem.getStreams() != 8 &&
            problem.getStreams() != 16) {
            candidates.insert(candidates.begin(), std::string(base) + "Generic");
        } else {
            candidates.emplace_back(std::string(base) + "Generic");
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
