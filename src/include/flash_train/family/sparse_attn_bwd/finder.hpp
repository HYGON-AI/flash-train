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
struct SparseAttnBwdRegistrationOrderFinder final {
    static const char* getName() { return "SparseAttnBwdRegistrationOrder"; }
    static std::vector<std::string> findCandidates(const SparseAttnBwdProblem& problem, const Constraints&) {
        // Only the BF16 baseline participates in this initial operator PR.
        // Add other precisions with their own records and support tests later.
        if (problem.getComputeType() != FTRAIN_NUMERIC_TYPE_BF16) { return {}; }
        // Packed/cached/split preferences were removed with those Primitives.
        // Future static selection ranks separately registered names using only
        // Problem metadata covered by getProblemKey(); never inspect device data
        // or add algorithm selection to executeImpl(). Keep a baseline fallback.
        return {"bf16_sparse_attn_bwd"};
    }
};

}  // namespace ftrain

#endif
