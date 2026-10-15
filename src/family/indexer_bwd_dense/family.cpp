// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/indexer_bwd_dense/bf16_indexer_bwd_dense.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense/fp16_indexer_bwd_dense.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense/fp32_indexer_bwd_dense.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_dense/mixed_fp32_indexer_bwd_dense.hpp"
#include "flash_train/family/indexer_bwd_dense/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<IndexerBwdDenseProblem>>> IndexerBwdDenseFamily::makeRecords() {
    return {std::make_shared<Bf16IndexerBwdDense>(), std::make_shared<Fp32IndexerBwdDense>(),
            std::make_shared<Fp16IndexerBwdDense>(), std::make_shared<MixedFp32IndexerBwdDense>()};
}

}  // namespace ftrain
