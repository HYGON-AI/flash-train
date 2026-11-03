// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_PROJ_RMS_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_PROJ_RMS_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/proj_rms_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct ProjRmsFwdRegistrationOrderFinder final {
    static const char* getName() { return "ProjRmsFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const ProjRmsFwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16ProjRmsFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16ProjRmsFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getXNumericType() != problem.getWeightNumericType()) ? "MixedFp32ProjRmsFwd"
                                                                                     : "Fp32ProjRmsFwd";
                break;
            default: return {};
        }
        const auto preferred = [&] {
            const auto m = problem.getRows(), n = problem.getOutputs(), k = problem.getReduction();
            const auto tiles    = (n + 15) / 16;
            const auto fragment = ((problem.getComputeType() == FTRAIN_NUMERIC_TYPE_BF16 ||
                                    problem.getComputeType() == FTRAIN_NUMERIC_TYPE_FP16)
                                       ? 16
                                       : 8);
            // gfx936 paired measurements: split-K also helps bandwidth-limited grids.
            if (n >= 16 && n <= 32 && k >= 16384 && k % 128 == 0) {
                if (problem.getComputeType() == FTRAIN_NUMERIC_TYPE_FP32 && m <= 32768) {
                    return std::string("Split8");
                }
                if (m <= 4096) return std::string("Split8");
                if (m <= 8192) return std::string("Split4");
                if (m <= 16384) return k >= 24576 ? std::string("Split4") : std::string("Split2");
            }
            int split = 1;
            if (((m + 63) / 64) * tiles * 4 < 64 && k >= 4096) {
                for (int s : {8, 4, 2}) {
                    if (k % s == 0 && (k / s) % fragment == 0) {
                        split = s;
                        break;
                    }
                }
            }
            switch (split) {
                case 8: return std::string("Split8");
                case 4: return std::string("Split4");
                case 2: return std::string("Split2");
                default: return std::string("");
            }
        }();
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "Split2", "Split4", "Split8"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
