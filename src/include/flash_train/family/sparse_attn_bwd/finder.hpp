// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_BWD_FINDER_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_BWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sparse_attn_bwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct SparseAttnBwdRegistrationOrderFinder final {
    static const char* getName() { return "SparseAttnBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseAttnBwdProblem& problem, const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16SparseAttnBwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16SparseAttnBwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKvNumericType()) ? "MixedFp32SparseAttnBwd"
                                                                                 : "Fp32SparseAttnBwd";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        const auto preferred = [&] {
            if (problem.getHeadDim() == 512 && (problem.getHeads() == 64 || problem.getHeads() == 128) &&
                problem.getTokens() >= 256 && problem.getKvTokens() % 8 == 0 &&
                (problem.getSelectionWidth() == 128 || problem.getSelectionWidth() == 160)) {
                return std::string("RegisterFused");
            }
            if ((problem.getHeadDim() == 512 || problem.getHeadDim() == 576) && problem.usesCachedBackward()) {
                return problem.usesLowMemoryWorkspace() ? std::string("CachedLowMemory") : std::string("Cached");
            }
            return (problem.getHeadDim() == 512 || problem.getHeadDim() == 576) ? std::string("Packed")
                                                                                : std::string("");
        }();
        // Refine the previously selected implementation into its old internal
        // schedule. Preserve the measured ordering among the original records.
        std::string schedule = preferred;
        const auto T = problem.getTokens(), H = problem.getHeads(), D = problem.getHeadDim();
        const auto S = problem.getKvTokens(), K = problem.getSelectionWidth();
        if (preferred == "Cached") {
            const bool split = D == 512 && (K >= 256 || (T >= 256 && K >= 128 && (H == 64 || H == 128)));
            const bool cached = (split && (H == 64 || H == 128)) ||
                                (D == 576 && problem.getValueDim() == 512 && K >= 128 &&
                                 (H == 8 || H == 16 || H == 32 || H == 64 || H == 128));
            if (!cached && problem.getValueDim() == D) {
                schedule = split ? "PackedSplit" : "Packed";
            } else if (D == 576 && problem.getValueDim() == 512) {
                if ((T == 4096 || T == 4097) && S == 4096 && K == 2048 && H == 64) {
                    schedule = "CachedPadded";
                } else if ((T == 4096 || T == 4097) && S == 4096 && K == 2048 && H == 32) {
                    schedule = "CachedGlmPair";
                } else if (H == 64 && T >= 2048 && K >= 1024 && K <= 8192 && S >= K && T * K <= 2147483647ULL) {
                    schedule = "CachedGlmLocal";
                }
            } else if (D == 512 && H == 64 && T >= 2048 && S >= K && K <= 8192 &&
                       T * K <= 2147483647ULL && (K == 128 || K == 160 || K >= 1024)) {
                if (T == 4096 && S == 5120 && K == 1152) {
                    schedule = problem.getTopkLengthAddress() ? "CachedCompact" : "CachedGrouped";
                } else {
                    schedule = S % 8 == 0 ? "CachedLocalAligned" : "CachedLocal";
                }
            } else if (D == 512 && H == 128 && K < 1024) {
                schedule = "CachedNoPrefetch";
            }
        } else if (preferred == "Packed" && D == 512 &&
                   (K >= 256 || (T >= 256 && K >= 128 && (H == 64 || H == 128)))) {
            schedule = "PackedSplit";
        }
        if (schedule == "Cached" && D == 576 && problem.getValueDim() == 512) schedule = "CachedGlm";
        if (K > 8192 && (schedule == "Cached" || schedule == "CachedNoPrefetch" || schedule == "CachedGlm")) {
            schedule += "GlobalIndices";
        }
        std::vector<std::string> candidates{std::string(base) + schedule};
        if (schedule != preferred) candidates.emplace_back(std::string(base) + preferred);
        for (const char* suffix :
             {"", "Packed", "Cached", "CachedLowMemory", "RegisterFused", "PackedSplit", "CachedNoPrefetch",
              "CachedLocal", "CachedLocalAligned", "CachedGrouped", "CachedCompact", "CachedGlmPair", "CachedGlmLocal",
              "CachedPadded", "CachedGlobalIndices", "CachedNoPrefetchGlobalIndices", "CachedGlm",
              "CachedGlmGlobalIndices"}) {
            if (preferred != suffix && schedule != suffix) candidates.emplace_back(std::string(base) + suffix);
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
