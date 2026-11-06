// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_CSA_COMPRESSOR_FWD_FINDER_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "flash_train/family/csa_compressor_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct CSACompressorFwdRegistrationOrderFinder final {
    static const char* getName() { return "CSACompressorFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const CSACompressorFwdProblem& problem, const Constraints&) {
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16 ||
            problem.getKvNumericType() != FTRAIN_NUMERIC_TYPE_BF16 ||
            problem.getScoreNumericType() != FTRAIN_NUMERIC_TYPE_BF16) {
            return {};
        }
        const char* base     = "CSACompressorFwd";
        const auto preferred = [&] {
            if (problem.getRatio() == 128 && (problem.getHeadDim() == 128 || problem.getHeadDim() == 512)) {
                return problem.getHeadDim() == 512 && problem.getCoff() == 1 ? std::string("R128WindowMajor")
                                                                             : std::string("R128");
            }
            if (problem.getRatio() == 128 && problem.getCoff() == 2 && problem.getHeadDim() >= 256 &&
                problem.getTotalCompressed() >= 64) {
                return std::string("");
            }
            return problem.getRatio() * problem.getCoff() <= 64 ? std::string("Cached") : std::string("Tiled");
        }();
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "Cached", "Tiled", "R128", "R128WindowMajor"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        // Preserve the established algorithm order, resolving its former internal schedule here.
        const auto algorithms = candidates;
        candidates.clear();
        const auto append = [&](const std::string& name) {
            if (std::find(candidates.begin(), candidates.end(), name) == candidates.end()) candidates.push_back(name);
        };
        const auto window  = problem.getRatio() * problem.getCoff();
        const auto resolve = [&](std::string name) {
            if (name == std::string(base) + "Cached") {
                const int capacity = window <= 4 ? 4 : window <= 8 ? 8 : window <= 16 ? 16 : window <= 32 ? 32 : 64;
                if (capacity != 4) name += "Capacity" + std::to_string(capacity);
            } else if (name == std::string(base) + "Tiled") {
                const int capacity = window <= 64 ? 64 : window <= 128 ? 128 : window <= 160 ? 160 : 256;
                if (capacity != 64) name += "Capacity" + std::to_string(capacity);
            }
            return name;
        };
        for (const auto& algorithm : algorithms) append(resolve(algorithm));
        for (const char* suffix : {"Cached", "Tiled", "CachedCapacity8", "CachedCapacity16", "CachedCapacity32",
                                   "CachedCapacity64", "TiledCapacity128", "TiledCapacity160", "TiledCapacity256",
                                   "TiledCapacity256Dim16", "TiledCapacity256Dim8", "TiledCapacity256Dim4"})
            append(std::string(base) + suffix);

        return candidates;
    }
};

}  // namespace ftrain

#endif
