// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/indexer_topk/fp32_indexer_topk.hpp"
#include "flash_train/family/indexer_topk/family.hpp"

namespace ftrain {

// A single implementation: the kernel's element types are fixed, so there is
// no dtype chain to separate.
std::vector<std::shared_ptr<const Primitive<IndexerTopKProblem>>> IndexerTopKFamily::makeRecords() {
    return {std::make_shared<Fp32IndexerTopK>()};
}

}  // namespace ftrain
