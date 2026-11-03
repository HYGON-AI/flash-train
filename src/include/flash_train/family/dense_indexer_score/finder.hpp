// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_INDEXER_SCORE_FINDER_HPP_
#define FTRAIN_FAMILY_DENSE_INDEXER_SCORE_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/dense_indexer_score/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct DenseIndexerScoreRegistrationOrderFinder final {
    static const char* getName() { return "DenseIndexerScoreRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const DenseIndexerScoreProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16DenseIndexerScore"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16DenseIndexerScore"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKNumericType() ||
                        problem.getQNumericType() != problem.getWNumericType())
                           ? "MixedFp32DenseIndexerScore"
                           : "Fp32DenseIndexerScore";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        const auto preferred =
            (problem.getHeadDim() == 128 &&
             (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64))
                ? ((problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64) && problem.getQueryLength() >= 512
                       ? std::string("Transposed")
                       : std::string("MultiQuery"))
                : ([&] {
                      if (problem.getQueryHeads() > 64 || problem.getHeadDim() > 128 ||
                          (problem.getQueryHeads() <= 4 && problem.getQueryLength() >= 4096)) {
                          return std::string("LargeD");
                      }
                      if (problem.getQueryHeads() <= 64 && problem.getQueryHeads() % 16 == 0 &&
                          (problem.getHeadDim() == 64 || problem.getHeadDim() == 128) &&
                          problem.getKeyLength() >= 256) {
                          return std::string("Hoisted");
                      }
                      return std::string("");
                  }());
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"MultiQuery", "Transposed", "", "LargeD", "Hoisted"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
