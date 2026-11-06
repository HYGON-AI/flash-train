// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_BWD_FINDER_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_BWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/proj_rms_compute_h_bwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct ProjRmsComputeHBwdRegistrationOrderFinder final {
    static const char* getName() { return "ProjRmsComputeHBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const ProjRmsComputeHBwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16ProjRmsComputeHBwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16ProjRmsComputeHBwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getXNumericType() != problem.getWeightNumericType()) ? "MixedFp32ProjRmsComputeHBwd"
                                                                                     : "Fp32ProjRmsComputeHBwd";
                break;
            default: return {};
        }
        const auto preferred =
            (problem.getOutputs() > 0 && problem.getOutputs() <= 32 && problem.getReduction() % 256 == 0)
                ? ((problem.getRows() >= 64 && problem.getOutputs() >= 16) ? std::string("MmacGradX")
                                                                           : std::string("Mmac"))
                : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"MmacGradX", "", "Mmac"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
