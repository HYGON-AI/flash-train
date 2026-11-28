// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_FINDER_HPP_
#define FTRAIN_FAMILY_SPARSE_INDEXER_PREDICT_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sparse_indexer_predict/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct SparseIndexerPredictRegistrationOrderFinder final {
    static const char* getName() { return "SparseIndexerPredictRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseIndexerPredictProblem& problem,
                                                   const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16SparseIndexerPredict"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16SparseIndexerPredict"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKNumericType() ||
                        problem.getQNumericType() != problem.getWNumericType())
                           ? "MixedFp32SparseIndexerPredict"
                           : "Fp32SparseIndexerPredict";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        auto preferred =
            (problem.getHeadDim() == 128 &&
             (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64) &&
             !problem.getTopkIndicesGlobal())
                ? ((problem.getQueryHeads() == 16 || problem.getQueryLength() < 512 ||
                    (problem.getQueryLength() < 4096 && problem.getKeyLength() < 512))
                       ? std::string("Single")
                       : ((problem.getQueryHeads() == 32 && problem.getSelectionWidth() == 2048)
                              ? std::string("SharedNormCache")
                              : std::string("Shared")))
                : ((problem.getHeadDim() == 64 || problem.getHeadDim() == 128 || problem.getHeadDim() == 256 ||
                    problem.getHeadDim() == 448 || problem.getHeadDim() == 512 || problem.getHeadDim() == 576)
                       ? std::string("GlobalQ")
                       : std::string(""));
        // D256 has only the MMAC-reduction implementation on gfx936/gfx938;
        // all other BF16 implementations are limited to D128. Device support
        // remains the primitive's responsibility; the filtered Plan is unchanged.
        // D64 showed no stable gain; D128 retains the faster Single/Shared policy.
        if (problem.getHeadDim() == 256) {
            preferred = "GlobalQMmacReduce";
        }
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"GlobalQMmacReduce", "Single", "Shared", "SharedNormCache", "", "GlobalQ"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
