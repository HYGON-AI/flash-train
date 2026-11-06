// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_ATTN_SCORE_FINDER_HPP_
#define FTRAIN_FAMILY_DENSE_ATTN_SCORE_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/dense_attn_score/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct DenseAttnScoreRegistrationOrderFinder final {
    static const char* getName() { return "DenseAttnScoreRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const DenseAttnScoreProblem& problem,
                                                   const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16DenseAttnScore"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16DenseAttnScore"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKNumericType()) ? "MixedFp32DenseAttnScore"
                                                                                : "Fp32DenseAttnScore";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        auto preferred = ((problem.getHeadDim() == 512 || problem.getHeadDim() == 576) &&
                          (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 ||
                           problem.getQueryHeads() == 64 || problem.getQueryHeads() == 128))
                             ? std::string("RingTeacher")
                             : ([&] {
                                   if (problem.getQueryHeads() > 64 || problem.getHeadDim() > 128) {
                                       return std::string("LargeD");
                                   }
                                   if (problem.getQueryHeads() <= 64 && problem.getQueryHeads() % 16 == 0 &&
                                       (problem.getHeadDim() == 64 || problem.getHeadDim() == 128) &&
                                       problem.getKeyLength() >= 128) {
                                       return std::string("Hoisted");
                                   }
                                   return std::string("");
                               }());
        // gfx936/gfx938 measurements favor bpermute for H64/D512, including
        // tail tiles. Primitive applicability retains the device support gate.
        if (problem.getQueryHeads() == 64 && problem.getHeadDim() == 512) {
            preferred = "RingTeacherBpermute";
        }
        std::vector<std::string> candidates{std::string(base) + preferred};
        // Padding also classifies tensor contents; a shape-only preference regresses unpadded inputs.
        for (const char* suffix : {"Padding", "RingTeacherBpermute", "RingTeacher", "", "LargeD", "Hoisted"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
