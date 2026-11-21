// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/sparse_attn_bwd/bf16_sparse_attn_bwd.hpp"
#include "flash_train/family/sparse_attn_bwd/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<SparseAttnBwdProblem>>> SparseAttnBwdFamily::makeRecords() {
    // Deferred optimized Primitives return as separate named records/files.
    // The Finder ranks records statically; kernels execute the configured record.
    // White-box tests discover this registry for name/clone/invalid-type checks;
    // numerical tests discover every applicable candidate through the Plan.
    // Initial operator PRs retain one precision only (BF16 by default).
    // FP16/FP32/mixed implementations remain outside this submission; add
    // their records and tests only when those implementations are introduced.
    return {std::make_shared<Bf16SparseAttnBwd>()};
}

}  // namespace ftrain
