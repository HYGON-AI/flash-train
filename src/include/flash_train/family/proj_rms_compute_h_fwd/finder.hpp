// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_COMPUTE_H_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/proj_rms_compute_h_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct ProjRmsComputeHFwdRegistrationOrderFinder final {
    static const char* getName() { return "ProjRmsComputeHFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const ProjRmsComputeHFwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16ProjRmsComputeHFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16ProjRmsComputeHFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getXNumericType() != problem.getWeightNumericType()) ? "MixedFp32ProjRmsComputeHFwd"
                                                                                     : "Fp32ProjRmsComputeHFwd";
                break;
            default: return {};
        }
        const auto preferred = [&] {
            const auto m = problem.getRows(), n = problem.getOutputs(), k = problem.getReduction();
            const auto tiles    = (n + 15) / 16;
            const auto fragment = 8;
            int split           = 1;
            if (((m + 63) / 64) * tiles * 4 < 64 && k >= 4096) {
                for (int s : {8, 4, 2}) {
                    if (k % s == 0 && (k / s) % fragment == 0) {
                        split = s;
                        break;
                    }
                }
            }
            if (m <= 1024 && tiles <= 2 && k >= 4096) {
                const int first = fragment == 8 && m > 512 ? 4 : 8;
                for (int s = first; s > 1; s /= 2) {
                    if (k % s == 0 && (k / s) % fragment == 0) {
                        split = s;
                        break;
                    }
                }
            }
            if (split == 1 && tiles <= 2) {
                return k % 64 == 0 ? std::string("V2ConflictFreeFullK") : std::string("V2ConflictFree");
            }
            switch (split) {
                case 8: return std::string("Split8");
                case 4: return std::string("Split4");
                case 2: return std::string("Split2");
                default: return std::string("");
            }
        }();
        std::vector<std::string> candidates{std::string(base) + preferred};
        // V2/V2FullK remain selectable: the conflict-free variants win the measured gfx936 cases.
        for (const char* suffix :
             {"V2FullK", "V2ConflictFreeFullK", "", "Split2", "Split4", "Split8", "V2", "V2ConflictFree"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
