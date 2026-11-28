// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_TARGET_FINDER_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_TARGET_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sparse_attn_target/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct SparseAttnTargetRegistrationOrderFinder final {
    static const char* getName() { return "SparseAttnTargetRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseAttnTargetProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16SparseAttnTarget"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16SparseAttnTarget"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKNumericType()) ? "MixedFp32SparseAttnTarget"
                                                                                : "Fp32SparseAttnTarget";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        const auto preferred =
            (problem.getBatch() == 1 && problem.getQueryLength() >= 1024 && problem.getHeadDim() == 512 &&
             (problem.getQueryHeads() == 64 || problem.getQueryHeads() == 128) &&
             (problem.getSelectionWidth() == 512 || problem.getSelectionWidth() == 1024 ||
              problem.getSelectionWidth() == 1152) &&
             problem.getKeyLength() * problem.getHeadDim() * 2 <= 2147483647ULL)
                ? std::string("Target512")
                : ((problem.getBatch() == 1 && problem.getQueryLength() == 4096 && problem.getKeyLength() == 4096 &&
                    problem.getHeadDim() == 576 && (problem.getQueryHeads() == 32 || problem.getQueryHeads() == 64) &&
                    problem.getSelectionWidth() == 2048)
                       ? std::string("Target576")
                       : ((problem.getHeadDim() == 64 || problem.getHeadDim() == 128 || problem.getHeadDim() == 256 ||
                           problem.getHeadDim() == 448 || problem.getHeadDim() == 512 || problem.getHeadDim() == 576)
                              ? std::string("GlobalQ")
                              : std::string("")));
        std::vector<std::string> candidates{std::string(base) + preferred};
        for (const char* suffix : {"", "GlobalQ", "Target512", "Target576"}) {
            if (preferred != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
