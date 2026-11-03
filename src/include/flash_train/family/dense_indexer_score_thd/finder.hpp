// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_INDEXER_SCORE_THD_FINDER_HPP_
#define FTRAIN_FAMILY_DENSE_INDEXER_SCORE_THD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/dense_indexer_score_thd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct DenseIndexerScoreThdRegistrationOrderFinder final {
    static const char* getName() { return "DenseIndexerScoreThdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const DenseIndexerScoreThdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16DenseIndexerScoreThd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16DenseIndexerScoreThd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKNumericType() ||
                        problem.getQNumericType() != problem.getWNumericType())
                           ? "MixedFp32DenseIndexerScoreThd"
                           : "Fp32DenseIndexerScoreThd";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        const auto preferred =
            (problem.getHeadDim() == 128 &&
             (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64))
                ? ((problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64) && problem.getTotalQ() >= 512
                       ? std::string("Transposed")
                       : std::string("MultiQuery"))
                : std::string("");
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "MultiQuery", "Transposed"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
