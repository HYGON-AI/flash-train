// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef FTRAIN_FAMILY_SPARSE_ATTN_BWD_FAMILY_HPP_
#define FTRAIN_FAMILY_SPARSE_ATTN_BWD_FAMILY_HPP_

#include <memory>
#include <vector>

#include "flash_train/family/sparse_attn_bwd/problem.hpp"
#include "flash_train/family/sparse_attn_bwd/finder.hpp"
#include "flash_train/primitive.hpp"
#include "flash_train/engine.hpp"

namespace ftrain {

// Everything the SparseAttnBwd family contributes to its engine.
struct SparseAttnBwdFamily {
    using Problem = SparseAttnBwdProblem;

    // The implementation catalog, defined in src/family/sparse_attn_bwd/family.cpp so
    // neither the Primitive list nor their includes grow this header.
    static std::vector<std::shared_ptr<const Primitive<Problem>>> makeRecords();
};

using SparseAttnBwdEngine = OpEngine<SparseAttnBwdFamily, SparseAttnBwdRegistrationOrderFinder>;

// Returns the process-wide SparseAttnBwd engine, constructed on the first
// call. Safe for concurrent calls: apart from its selection cache the engine
// never changes after construction.
const SparseAttnBwdEngine& sparseAttnBwdEngine();

}  // namespace ftrain

#endif
