// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_INDEXER_BWD_SPARSE_FINDER_HPP_
#define FTRAIN_FAMILY_INDEXER_BWD_SPARSE_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/indexer_bwd_sparse/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct IndexerBwdSparseRegistrationOrderFinder final {
    static const char* getName() { return "IndexerBwdSparseRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const IndexerBwdSparseProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16IndexerBwdSparse"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16IndexerBwdSparse"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getIndexQNumericType() != problem.getIndexKNumericType() ||
                        problem.getIndexQNumericType() != problem.getWeightsNumericType())
                           ? "MixedFp32IndexerBwdSparse"
                           : "Fp32IndexerBwdSparse";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        const auto preferred =
            (problem.getHeadDim() == 128 && problem.getHeads() == 64 && problem.getSelectionWidth() == 2048 &&
             problem.getQueryLength() == 4096 && problem.getKeyLength() == 4096 && !problem.getTopkIndicesGlobal())
                ? std::string("PaddedBufferCast")
                : ([&] {
                      auto algorithm = std::string("");
                      if (problem.getHeadDim() == 128 && problem.getHeads() == 32 &&
                          problem.getSelectionWidth() == 2048 && problem.getQueryLength() == 4096 &&
                          problem.getKeyLength() == 4096) {
                          algorithm = std::string("CachedScores");
                      } else if (problem.getHeadDim() == 128 && problem.getHeads() == 64 &&
                                 ((problem.getSelectionWidth() == 1024 && problem.getQueryLength() >= 2048) ||
                                  (problem.getSelectionWidth() == 2048 && problem.getQueryLength() == 4096 &&
                                   problem.getKeyLength() == 4096))) {
                          algorithm = std::string("TwoWave");
                      } else if (problem.getHeadDim() == 128 && problem.getHeads() == 64 &&
                                 problem.getSelectionWidth() % 16 == 0) {
                          algorithm = std::string("Aligned");
                      }
                      if (problem.getBatch() * problem.getKeyLength() * problem.getHeadDim() >= (1u << 17)) {
                          algorithm = algorithm.empty() ? "GenericCast" : algorithm + "Cast";
                      }
                      return algorithm;
                  }());
        std::vector<std::string> candidates{std::string(base) + preferred};
        // On gfx936 the flat padded path trails the buffer path at B=1/2/4.
        // Keep it available for explicit selection, including spans the buffer path rejects.
        for (const char* suffix : {"PaddedBufferCast", "PaddedFlatCast", "", "GenericCast", "Aligned", "AlignedCast",
                                   "CachedScores", "CachedScoresCast", "TwoWave", "TwoWaveCast"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
