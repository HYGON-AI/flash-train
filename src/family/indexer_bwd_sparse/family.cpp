// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <vector>

#include "flash_train/primitive/hygon/indexer_bwd_sparse/bf16_indexer_bwd_sparse.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_sparse/fp16_indexer_bwd_sparse.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_sparse/fp32_indexer_bwd_sparse.hpp"
#include "flash_train/primitive/hygon/indexer_bwd_sparse/mixed_fp32_indexer_bwd_sparse.hpp"
#include "flash_train/family/indexer_bwd_sparse/family.hpp"

namespace ftrain {

// Registered implementations are filtered by the requested compute type.
std::vector<std::shared_ptr<const Primitive<IndexerBwdSparseProblem>>> IndexerBwdSparseFamily::makeRecords() {
    return {std::make_shared<Bf16IndexerBwdSparse>(), std::make_shared<Fp32IndexerBwdSparse>(),
            std::make_shared<Fp16IndexerBwdSparse>(), std::make_shared<MixedFp32IndexerBwdSparse>()};
}

}  // namespace ftrain
