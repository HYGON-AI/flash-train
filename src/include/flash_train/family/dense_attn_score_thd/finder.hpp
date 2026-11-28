// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_DENSE_ATTN_SCORE_THD_FINDER_HPP_
#define FTRAIN_FAMILY_DENSE_ATTN_SCORE_THD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/dense_attn_score_thd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct DenseAttnScoreThdRegistrationOrderFinder final {
    static const char* getName() { return "DenseAttnScoreThdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const DenseAttnScoreThdProblem& problem,
                                                   const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16DenseAttnScoreThd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16DenseAttnScoreThd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKNumericType()) ? "MixedFp32DenseAttnScoreThd"
                                                                                : "Fp32DenseAttnScoreThd";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        auto preferred = ((problem.getHeadDim() == 512 || problem.getHeadDim() == 576) &&
                          (problem.getQueryHeads() == 16 || problem.getQueryHeads() == 32 ||
                           problem.getQueryHeads() == 64 || problem.getQueryHeads() == 128))
                             ? std::string("RingTeacher")
                             : std::string("");
        // Use the gfx938-measured order for H64/D512, including ragged
        // segments and tail tiles. Primitives retain device support checks.
        if (problem.getQueryHeads() == 64 && problem.getHeadDim() == 512) {
            return {std::string(base) + "RingTeacherBpermute", std::string(base) + "RingTeacher", base};
        }
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"RingTeacherBpermute", "", "RingTeacher"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
