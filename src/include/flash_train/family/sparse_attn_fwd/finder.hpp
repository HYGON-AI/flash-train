// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_FWD_FINDER_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_FWD_FINDER_HPP_

#include <string>
#include <vector>

#include "flash_train/family/sparse_attn_fwd/problem.hpp"
#include "flash_train/constraints.hpp"

namespace ftrain {

// Rank fixed primitives within the input precision chain; isApplicable checks shape and device support.
// The historical Finder name remains stable for environment controls.
struct SparseAttnFwdRegistrationOrderFinder final {
    static const char* getName() { return "SparseAttnFwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseAttnFwdProblem& problem,
                                                   const Constraints&) {
        const char* base = nullptr;
        switch (problem.getComputeType()) {
            case FTRAIN_NUMERIC_TYPE_BF16: base = "Bf16SparseAttnFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP16: base = "Fp16SparseAttnFwd"; break;
            case FTRAIN_NUMERIC_TYPE_FP32:
                base = (problem.getQNumericType() != problem.getKvNumericType()) ? "MixedFp32SparseAttnFwd"
                                                                                 : "Fp32SparseAttnFwd";
                break;
            default: return {};
        }
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {base}; }
        std::string partial_alternative;
        const auto preferred = [&] {
            // Use the gfx938-measured threshold for both length-port modes.
            // Device/ISA support remains the primitive's responsibility.
            if (problem.getHeads() > 0 && problem.getHeads() < 64 && problem.getHeadDim() == 576 &&
                problem.getSelectionWidth() == 2048) {
                if (problem.getHeads() == 8 || problem.getHeads() == 16 || problem.getHeads() == 32) {
                    const bool four_waves = problem.getTokens() <= 72;
                    partial_alternative = four_waves ? "MlaPartial2" : "MlaPartial4";
                    return four_waves ? std::string("MlaPartial4") : std::string("MlaPartial2");
                }
                return std::string("MlaPartial4");
            }
            if (problem.getTokens() < 1024) { return std::string(""); }
            if ((problem.getHeads() == 64 || problem.getHeads() == 128) && problem.getHeadDim() == 512 &&
                (problem.getSelectionWidth() == 128 || problem.getSelectionWidth() == 160)) {
                return std::string("ShortTwoPass");
            }
            if ((problem.getHeads() == 64 || problem.getHeads() == 128) && problem.getHeadDim() == 512 &&
                problem.getSelectionWidth() == 256 && problem.getTopkLengthAddress() == nullptr &&
                problem.getTokens() % 128 == 0 &&
                problem.getKvTokens() == problem.getTokens() + problem.getTokens() / 128) {
                return std::string("CsaPadded");
            }
            if ((problem.getHeads() == 64 || problem.getHeads() == 128) && problem.getHeadDim() == 512 &&
                problem.getSelectionWidth() == 256 && problem.getTopkLengthAddress() == nullptr) {
                return std::string("MlaTail256");
            }
            if ((problem.getHeads() == 64 || problem.getHeads() == 128) && problem.getHeadDim() == 512 &&
                (problem.getSelectionWidth() == 1024 || problem.getSelectionWidth() == 1152 ||
                 problem.getSelectionWidth() == 2048)) {
                return std::string("MlaMasked");
            }
            if ((problem.getHeads() == 64 || problem.getHeads() == 128) &&
                ((problem.getHeadDim() == 512 &&
                  (problem.getSelectionWidth() == 128 || problem.getSelectionWidth() == 256 ||
                   problem.getSelectionWidth() == 512 || problem.getSelectionWidth() == 640 ||
                   problem.getSelectionWidth() == 1024 || problem.getSelectionWidth() == 1152 ||
                   problem.getSelectionWidth() == 2048)) ||
                 (problem.getHeadDim() == 576 && problem.getSelectionWidth() == 2048))) {
                return std::string("Mla64");
            }
            return std::string("");
        }();
        std::vector<std::string> candidates{std::string(base) + preferred};
        if (!partial_alternative.empty()) candidates.emplace_back(std::string(base) + partial_alternative);
        // Duplicate576 classifies tensor contents; random H64/D576 inputs regress.
        // MlaTail160 is retained for tuning; ShortTwoPass wins its measured shapes.
        for (const char* suffix : {"Duplicate576", "", "Mla64", "MlaMasked", "MlaTail160", "MlaTail256", "MlaPartial2",
                                   "MlaPartial4", "ShortTwoPass", "CsaPadded"}) {
            if (preferred != suffix && (partial_alternative.empty() || partial_alternative != suffix)) {
                candidates.emplace_back(std::string(base) + suffix);
            }
        }
        return candidates;
    }
};

}  // namespace ftrain

#endif
