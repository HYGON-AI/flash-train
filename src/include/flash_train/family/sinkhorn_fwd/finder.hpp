// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SINKHORN_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_SINKHORN_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sinkhorn_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct SinkhornFwdRegistrationOrderFinder final {
    static const char* getName() { return "SinkhornFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SinkhornFwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16SinkhornFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16SinkhornFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32: base = "Fp32SinkhornFwd"; break;
            default: return {};
        }
        const auto preferred =
            (problem.getMatrixSize() == 8 ||
             (problem.getMatrixSize() == 4 && (problem.getBatch() >= 1024 || problem.getNumIterations() >= 10)))
                ? std::string("Wave")
                : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "Wave"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
