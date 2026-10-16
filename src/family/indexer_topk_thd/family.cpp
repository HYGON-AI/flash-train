// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/indexer_topk_thd/fp32_indexer_topk_thd.hpp"
#include "flash_train/family/indexer_topk_thd/family.hpp"

namespace ftrain {

// A single implementation: fixed INT32 out, FP32 in.
std::vector<std::shared_ptr<const Primitive<IndexerTopKThdProblem>>> IndexerTopKThdFamily::makeRecords() {
    return {std::make_shared<Fp32IndexerTopKThd>()};
}

}  // namespace ftrain
