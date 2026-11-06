// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_BWD_DENSE_FINDER_HPP_
#define FTRAIN_FAMILY_INDEXER_BWD_DENSE_FINDER_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "flash_train/family/indexer_bwd_dense/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct IndexerBwdDenseRegistrationOrderFinder final {
    static const char* getName() { return "IndexerBwdDenseRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const IndexerBwdDenseProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16IndexerBwdDense"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16IndexerBwdDense"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getIndexQNumericType() != problem.getIndexKNumericType() ||
                        problem.getIndexQNumericType() != problem.getWeightsNumericType())
                           ? "MixedFp32IndexerBwdDense"
                           : "Fp32IndexerBwdDense";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        const auto preferred = (problem.getHeads() == 64 && problem.getHeadDim() == 128 && problem.getRatio() > 0 &&
                                problem.getQueryLength() >= 512)
                                   ? std::string("TwoWave")
                                   : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "TwoWave"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        // Preserve the established algorithm order, resolving its former internal schedule here.
        const auto algorithms = candidates;
        candidates.clear();
        const auto append = [&](const std::string& name) {
            if (std::find(candidates.begin(), candidates.end(), name) == candidates.end()) candidates.push_back(name);
        };
        const auto resolve = [&](std::string name) {
            if (name != std::string(base) + "TwoWave") return name;
            if (problem.getRatio() > 1 && problem.getRatio() % 2 == 0) return name + "Pair";
            if (problem.getRatio() == 1 && problem.getKeyLength() >= 2048 && problem.hasUnitSmScale() &&
                problem.getBatch() * problem.getKeyLength() * 128ULL * 2 >=
                    (problem.getBatch() * problem.getQueryLength() + 16) * sizeof(int))
                return name + "Proof";
            return name;
        };
        for (const auto& algorithm : algorithms) append(resolve(algorithm));
        append(std::string(base) + "TwoWave");
        append(std::string(base) + "TwoWavePair");
        append(std::string(base) + "TwoWaveProof");

        return candidates;
    }
};

}  // namespace ftrain

#endif
