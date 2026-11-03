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
    return {std::make_shared<Fp32IndexerTopKOrderedUInt>(),
            std::make_shared<Fp32IndexerTopKGroupedAll>(),
            std::make_shared<Fp32IndexerTopKGroupedRadix>(),
            std::make_shared<Fp32IndexerTopK>(),
            std::make_shared<Fp32IndexerTopKSize4>(),
            std::make_shared<Fp32IndexerTopKSize8>(),
            std::make_shared<Fp32IndexerTopKSize16>(),
            std::make_shared<Fp32IndexerTopKSize32>(),
            std::make_shared<Fp32IndexerTopKSize64>(),
            std::make_shared<Fp32IndexerTopKSize128>(),
            std::make_shared<Fp32IndexerTopKSize256>(),
            std::make_shared<Fp32IndexerTopKSize512>(),
            std::make_shared<Fp32IndexerTopKWideRadixCapacity16>(),
            std::make_shared<Fp32IndexerTopKWideRadixCapacity32>(),
            std::make_shared<Fp32IndexerTopKWideRadixCapacity64>(),
            std::make_shared<Fp32IndexerTopKWideRadix>(),
            std::make_shared<Fp32IndexerTopKRadix1024>()};
}

}  // namespace ftrain
